import PerceptualSimilarity as ps
import PerceptualSimilarity.models.dist_model as dm
import PerceptualSimilarity.util.util as psutil
import torch


def get_perceptual_distance(input_image_list,output_image_list,model_path=None):
    
    model = dm.DistModel()
    if model_path is None:
        model.initialize(model = 'net-lin', net='alex', model_path ='LPIPS/PerceptualSimilarity/weights/v0.1/alex.pth', use_gpu=True)
    else:
        model.initialize(model = 'net-lin', net='alex', model_path = model_path, use_gpu=True)
    
    dist_scores = np.zeros((len(input_image_list),len(output_image_list)))
    for i,img_i in enumerate(input_image_list):
        for j,img_o in enumerate(output_image_list):
            ex_i = psutil.im2tensor(psutil.load_image(img_i))
            ex_o = psutil.im2tensor(psutil.load_image(img_o))
            dist[i,j]=model.forward(ex_i,ex_o)[0]

    return dist,np.mean(dist)
