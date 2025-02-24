from llava.train.train import train
# from llava.train.train_get_lastbatch import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
