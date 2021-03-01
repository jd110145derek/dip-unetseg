from model import *
from data import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# myGene = trainGenerator(2,'./data/train3_all','image','label',data_gen_args,save_to_dir = None)

# model = unet()
# model_checkpoint = ModelCheckpoint('model_3.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit(myGene,steps_per_epoch=103,epochs=10,callbacks=[model_checkpoint])


model = load_model("./model_3.hdf5")
testGene = testGenerator("./data/test3/0.jpg")
results = model.predict(testGene,steps=1,verbose=1)
saveResult("./data/test3",results)
