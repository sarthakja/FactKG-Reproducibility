from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pickle
from tqdm import tqdm
import argparse
import time

#This method defines the command line arguments
def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_data_path', required=True, type=str)
    parser.add_argument('--model_name', default="google/flan-t5-xl", type=str)
    args = parser.parse_args()
    return args

#This main method gives the test accuracy, in this code there is no training involved since Flan-T5 is being used in a zero-shot setting
def main(args):
    model_name = args.model_name    
    valid_data_path = args.valid_data_path

    with open(valid_data_path, 'rb') as pickle_file:
        valid_pickle = pickle.load(pickle_file)

    ###If there is cuda memory error, change "google/flan-t5-xl" to "google/flan-t5-large" or "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda')

    # the following 2 hyperparameters are task-specific
    max_source_length = 256
    max_target_length = 16

    # encode the inputs
    task_prefix = "Is this claim True or False? Claim: "
    key_list = list(valid_pickle)#Get the list of claims

    tot_corr = 0
    tot_num = 0

    
    k_ = 0
    input_sequences = []
    #claim_types stores the different claim type names so as to calulate claim wise accuracy
    claim_types=['num1', 'multi claim', 'existence', 'multi hop', 'negation']
    claim_corrects=[0,0,0,0,0]
    t1=time.time()
    #This for loop first builds a batch of claims(the first 3 if conditions, and batch size is 64), and then evaluates that batch
    for key in tqdm(key_list):
        #print("Entered for loop")
        #This first if condition starts building the batch
        if k_ % 64 == 0:
            input_sequences = [key]
            k_ += 1
            continue
        #This second elif condition builds the items in the batch that are indexed from 1 to 62 
        elif k_ % 64 in [i+1 for i in range(62)]:
            input_sequences.append(key)
            if(k_ != (len(key_list)-1)):
              #print("Continuing",k_)
              k_ += 1
              continue

        #This third elif condition builds the last element in the batch   
        elif k_ % 64 == 63 or key == key_list[-1]:
            input_sequences.append(key)
            #print("Printing k_", k_)
            k_ += 1

        encoding = tokenizer(
                [task_prefix + input_sequences[i] for i in range(len(input_sequences))],
                padding="longest",
                max_length=max_source_length,
                truncation=True,
                return_tensors="pt",
            ).to('cuda')

        outputs = model.generate(encoding.input_ids)

        #This for loop calculates the no of claims classified correctly by the Flan-T5 model 
        for i in range(outputs.shape[0]):
                tot_num += 1
                tmp_decode = tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                tmp_target = valid_pickle[input_sequences[i]]
                tmp_types = tmp_target['types']

                #The 2 if conditions are for the 2 cases when a claim is classified correctly
                if True in tmp_target['Label'] and ('True' in tmp_decode or 'true' in tmp_decode or 'yes' in tmp_decode or 'Yes' in tmp_decode) and ('False' not in tmp_decode and 'false' not in tmp_decode and 'no' not in tmp_decode and 'No' not in tmp_decode):
                    tot_corr += 1
                    for typ in tmp_types:
                      if(typ in claim_types):
                        claim_corrects[claim_types.index(typ)]+=1
                if False in tmp_target['Label'] and ('False' in tmp_decode or 'false' in tmp_decode or 'no' in tmp_decode or 'No' in tmp_decode) and ('True' not in tmp_decode and 'true' not in tmp_decode and 'yes' not in tmp_decode and 'Yes' not in tmp_decode):
                    tot_corr += 1
                    for typ in tmp_types:
                      if(typ in claim_types):
                        claim_corrects[claim_types.index(typ)]+=1

    #Printing the total accuracy and accuracy by claim type
    print('Total num is ', tot_num)
    print('Accuracy: ', tot_corr/tot_num)
    print("Length of dataset: ", len(list(valid_pickle)))
    validationTypesCounts = [1914, 3069, 870, 1874, 1314]
    for i in range(5):
      typacc = claim_corrects[i]/validationTypesCounts[i]
      print("Accuracy for type: ", claim_types[i], " is: ", typacc)

    t2=time.time()
    print("Time taken to validate: ", t2-t1)

if __name__ == '__main__':
    args = define_argparser()
    main(args)