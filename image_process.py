import process_utils
import torchvision
device = 'cuda'
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()
process_utils.apply_mask_rcnn(im_path='coffeechat.jpg', model=model, device=device, topk=10)