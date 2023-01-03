import random
import ads3 as ads3

labels = ads3.get_labels()

train_file = "data/train.txt"

train_final = "data/train_final.txt"
valid_final = "data/valid_final.txt"

with open(train_file) as f:
    lines = f.readlines()

    f_train = open(train_final, "w")
    f_valid = open(valid_final, "w")

    for line in lines:
        label = line.strip().split("/")[1]
        label = labels.index(label)

        im_path = "image/" + line.strip()

        rand = random.random()

        # create 70-30 split
        if rand < 0.7:
            # is train
            f_train.write(f"{im_path} {label}\n")
        else: 
            # is valid
            f_valid.write(f"{im_path} {label}\n")
        
    f_train.close()
    f_valid.close()


        
    
