from cv2 import ellipse2Poly
from src.data_loader.youtube_loader import YTB_DB
from src.constants import YOUTUBE_DATA
from tqdm import tqdm

def try_data(data):
    print("Iterating through all valid samples!")
    counter =0
    for idx in tqdm(range(len(data))):
        try:
            s = data[idx]
        except Exception as e:
            counter+=1
    if counter==0:
        print("Everything seems fine, the valid-invalid csv is upto date!")
    else:
        print(f"HOOPLA!  Unable to read {counter} samples")
def main():
   
    
    print("Test data")
    data = YTB_DB(YOUTUBE_DATA, split="test")
    try_data(data)
  
    print("Val data")
    data = YTB_DB(YOUTUBE_DATA, split="val")
    try_data(data)

    print("Train data")
    data = YTB_DB(YOUTUBE_DATA)
    try_data(data)

main()



