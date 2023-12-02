# This file preprocesses the training, validation, and test sets for the prediction of claims
# as supported/unsupported in the test set

import json
import pickle as pkl
from random import choice
from itertools import permutations, chain
from termcolor import colored
from tqdm.auto import tqdm
import os

#This class defines methods for processing a knowledge graph and extracting information from it.
#Details about each method are provided above the method
class KG():
    def __init__(self, kg):
        super().__init__()
        self.kg = kg
        
    #This method is for a particular claim. ents is the list of entities in the claim. rels is the dictio-
    #nary corresponding to claim['Evidence']
    #The method searches for 2 types of paths in the Knowledge graph for each entity:
    #The first path starts at an entity in the calim and ends at another entity in the claim itself
    #The second path starts at an entity in the claim, but may end at an entity not present in the claim
    
    # The method returns a dictionary of 2 keys: "connected" and "walkable" respectively
    # The value of "connected" is a list containing the paths of first type
    # The value of "walkable" is a list containing the paths of second type
    def search(self, ents, rels):
        connected = list() #List of paths where each path is a list of form [entity->rel->entity1->...->last_entity in path]
                           #where last_entity is an entity in claim different from the first entity in the path
        walkable = list() # Same as connected above, with the only difference that the last entity may
                          # not be an entity in claim 
        seen = dict()
        
        for e in ents:
            if e in rels:
                for path in rels[e]:
                    leaf = ents[:]
                    leaf.remove(e)
                    result = self.walk(start=e, path=path, ends=leaf)
                    #result is the list of entities obtaines by following the path. 
                    #result[1] is the path where last entity is in leaf. result[0]'s
                    #last entity may or may not be in leaf 
                    if result != (None, None):
                        if result[1] is not None:
                            query = str(sorted([result[1][0], result[1][-1]]))
                            if query not in seen:
                                #conn_with_rel is a list: [entity->rel->entity1->...->last_entity in path]
                                conn_with_rel = result[1][:1]+list(chain(*[[r, e] for r, e in zip(path, result[1][1:])]))
                                connected.append(conn_with_rel)
                                seen[query] = None
                        if result[0][0] != result[0][-1]:
                            query = str(sorted([result[0][0], result[0][-1]]))
                            if query not in seen:
                                # walk_with_rel is similar to conn_with_rel
                                walk_with_rel = result[0][:1]+list(chain(*[[r, e] for r, e in zip(path, result[0][1:])]))
                                walkable.append(walk_with_rel)
                                seen[query] = None
                
        return {"connected":connected, "walkable":walkable}

    # This method performs a walk starting from the entity in the "start" argument along the path
    # in the path argument.
    #The method returns 2 types of paths: 
    #The first path starts from the entity in "start" and ends at another entity in the claim itself
    #The second path starts from the entity in "start", but may end at an entity not present in the claim

    # The method may return "None,None" if no path is found, "path_type_2, None" if there is no path
    # that ends at an entity in the claim, or "path_type_2, path_type_1" if there is a path that 
    # ends at an entity in the claim            
    def walk(self, start, path, ends=None):
        branches = [[start,],]
        for r in path:
            updated_branches = list()
            for branch in branches:
                h = branch[-1]
                ts = self.get_tail(h, r)
                if (r == path[-1]) and ts:
                    rand_branch = branch+[choice(list(ts.keys())),]
                    for e in ends:
                        if e in ts:
                            # print("printing")
                            # print(rand_branch, branch+[e,])
                            return rand_branch, branch+[e,]
                    return rand_branch, None
                else:
                    if ts:
                        for t in ts:
                            updated_branches.append(branch+[t,])
            if len(updated_branches) <= len(branches):
                return None, None
            branches = updated_branches

    # This method takes an entity (the argument h), a relation(the argument r) and returns a
    # dictionary where keys are the entities connected to h by the relation r and their values is None
    def get_tail(self, h, r):
        if h in self.kg:
            if r in self.kg[h]:
                return {x:None for x in self.kg[h][r]}
            else:
                return {}
        else:
            return {}    

# This method uses the KG class above to create the dictionary as returned by the search method
# of the KG class for each claim. The created dictionary is saved in the current directory 
def prepare_input(data_path, kg_path):
    predicted_rs = list()

    with open("../retrieve/model/relation_predict/test_relations_top3.json") as jsf:
        js = json.load(jsf)

    #predicted_rs is a list of tuples of form: (index, claim, predicted relation output)        
    for idx in js["claims"]:
        predicted_rs.append((idx, js["claims"][idx], js["output"][idx])) 

    predicted_hops = list()

    with open("../retrieve/model/hop_predict/predictions_hop.json") as jsf:
        js = json.load(jsf)

    #print("Here at 95")
    # predicted_hops is a list of tuples of form: (index, claim, predicted hop output)        
    for idx in tqdm(js["claims"]):
        predicted_hops.append((idx, js["claims"][idx], js["predict"][idx])) 
    
    #Load the knowledge graph
    with open(kg_path, "rb") as pkf:
        raw_kg = pkl.load(pkf)
    kg = KG(raw_kg)

    search_results = dict()

    #Load the training dataset
    with open(os.path.join(data_path, 'factkg_train.pickle'), "rb") as pkf:
        db = pkl.load(pkf)

    #print("Here at 109")
    #each claim key in search_results contains a dictionary of 2 keys: connected paths and walkable paths
    #This is for training set
    for i, (claim, elem) in tqdm(enumerate(db.items()), total=len(db)):
        ents = elem["Entity_set"]
        rels = elem["Evidence"]
        search_results[claim] = kg.search(ents, rels)
            
    assert len(search_results)==len(db)

    # Save the dictionary(the description of the dictionary is given above the function) for training set 
    with open("./train_candid_paths.bin", "wb") as pkf:
        pkl.dump(search_results, pkf)

    #Load the development set
    with open(os.path.join(data_path, 'factkg_dev.pickle'), "rb") as pkf:
        db = pkl.load(pkf)
    kg_dev = KG(raw_kg)

    search_results = dict()

    #each claim key in search_results contains a dictionary of 2 keys: connected paths and walkable paths
    #This is for validation set
    for i, (claim, elem) in tqdm(enumerate(db.items()), total=len(db)):
        ents = elem["Entity_set"]
        rels = elem["Evidence"]
        search_results[claim] = kg_dev.search(ents, rels)
            
    assert len(search_results)==len(db)

    # Save the dictionary(the description of the dictionary is given above the function) for validation set 
    with open("./dev_candid_paths.bin", "wb") as pkf:
        pkl.dump(search_results, pkf)

    predicted_rs = dict()

    # Load the the predicted relations(the relations predicted when main.py in relation_predict was run)
    # for the test set
    with open("../retrieve/model/relation_predict/test_relations_top3.json") as jsf:
        js = json.load(jsf)

    #predicted_rs is a dictionary: claim-> predicted relation output    
    for idx in js["claims"]:
        predicted_rs[js["claims"][idx]] =  js["output"][idx]

    predicted_hops = dict()

    # Load the the predicted hops(the relations predicted when main.py in hop_predict was run)
    # for the test set
    with open("../retrieve/model/hop_predict/predictions_hop.json") as jsf:
        js = json.load(jsf)
            
    #predicted_hops is a dictionary: claim-> predicted hop output
    for idx in js["claims"]:
        predicted_hops[js["claims"][idx]] =  js["predict"][idx]
        
    #Load the test set    
    with open(os.path.join(data_path, 'factkg_test.pickle'), "rb") as pkf:
        db = pkl.load(pkf)
    kg_test = KG(raw_kg)


    search_results = dict()

    #each claim key in search_results contains a dictionary of 2 keys: connected paths and walkable paths
    #This is for test set
    #To generate the search_results dict, kg_test.search method is used.
    #The argument rels is slightly different than how it was for train and validation sets
    #Here rels is same for all entities in the claim and is all possible permutations of paths whose
    #length is of predicted hop size 
    for i, elem in tqdm(enumerate(db.items()), total=len(db)):
        ents = elem[-1]["Entity_set"]
        candids = predicted_rs[elem[0]]
        hop = predicted_hops[elem[0]]
        claim = elem[0]
        
        rels = {e:list(permutations(candids, r=hop)) for e in ents}
        
        search_results[claim] = kg_test.search(ents, rels)

    assert len(search_results)==len(db)
    # Save the dictionary(the description of the dictionary is given above the function) for test set 
    with open("./test_candid_paths_top3.bin", "wb") as pkf:
        pkl.dump(search_results, pkf)