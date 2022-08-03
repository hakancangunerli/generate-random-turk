from predict import predict_xx_xy
import trnamegen as trn
import requests
import cv2
import shutil

def main():
        # TODO: come up with a way to make the names more "modern"
    # change this gender stuff to sex or sth 
    # prettify code 
    # generate a couple of people at a time
    
    url = "https://thispersondoesnotexist.com/image"
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open("img.png", 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

    result = predict_xx_xy("img.png")

    # randomly generate a name using trnamegen 
    # my god these names are BAD
    if result == ("Male"):
        randomName = trn.randomName(1)
    else:
        randomName = trn.randomName(2)

    # standardize the letters since cv2 doesn't have weird letter support
    def tr_to_std_latin(randomName):
        randomName = randomName.lower()
        randomName = randomName.replace('ü', 'u')
        randomName = randomName.replace('ç','c')
        randomName = randomName.replace('ı','i')
        randomName = randomName.replace('ğ','g')
        randomName = randomName.replace('ö','o')
        randomName = randomName = randomName.replace('ş','s')
        randomName = randomName.title()
        return randomName
        

    rando = tr_to_std_latin(randomName)
    im = cv2.imread('img.png')
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(im, rando, (10, 100), font, 2, (0,0,255), 2, cv2.LINE_AA) 
    cv2.imwrite('img.png', im)