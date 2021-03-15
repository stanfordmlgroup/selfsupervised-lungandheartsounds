def generate_heart_disjoint_pretrain():
    train_set = set()
    with open('splits/train.txt', 'r') as train_list:
        for train_example in train_list:
            #print(train_example)
            train_set.add(train_example)

    pretrain_only = open("splits/pretrain-only.txt","w")
    with open('splits/pretrain.txt', 'r') as pretrain_list:
        for pretrain_candidate in pretrain_list:
            #print(pretrain_candidate)
            if pretrain_candidate not in train_set:
                pretrain_only.write(pretrain_candidate)
                #pretrain_only.write("\n")
    pretrain_only.close()

if __name__ == "__main__":
    generate_heart_disjoint_pretrain()
