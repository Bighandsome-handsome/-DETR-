import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from models import deformable_detr  # 改成deformable_detr

DEFAULT_IMG = "/root/Deformable-DETR-mainversion4/tile_0027.png"
DEFAULT_CKPT = '/root/Deformable-DETR-mainversion4/exps/r50_deformable_detr/checkpoint0099.pth'
DEFAULT_OUT = '/root/Deformable-DETR-mainversion4/output/result.png'

def load_model_from_checkpoint(checkpoint_path, device):
    # 允许反序列化 argparse.Namespace
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt['args']
    args.device = device
    
    # 构建模型
    model, _, _ = deformable_detr.build(args)  # 改成deformable_detr
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    print("✅ 模型加载成功！")
    return model


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), img


def postprocess(outputs, img, threshold=0.20):  # SAR船舶阈值调高点
    """将Deformable DETR输出转换为像素坐标"""
    # 取最后一层的输出
    logits = outputs['pred_logits'][0]  
    boxes = outputs['pred_boxes'][0]
    
    # 计算概率（只有ship一个类）
    probs = logits.sigmoid()  # [num_queries, 1]
    scores = probs.squeeze(-1)  # [num_queries]
    
    # 过滤低分框
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    
    # 转换坐标：cx,cy,w,h -> x1,y1,x2,y2
    w, h = img.size
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * w  # x1
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * h  # y1
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]/2) * w  # x2
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]/2) * h  # y2
    
    return boxes_xyxy.cpu().numpy(), scores.cpu().numpy()


def plot_results(img, boxes, scores, save_path="result.jpg"):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    ax = plt.gca()
    
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'Ship: {score:.2f}', 
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=10, color='white')
    
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 检测到 {len(boxes)} 个目标，结果保存到 {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default=DEFAULT_IMG, help='path to input image')
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT, help='path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.20, help='confidence threshold')
    parser.add_argument('--out', default=DEFAULT_OUT, help='path to save output image')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # 预处理
    img_tensor, img = preprocess(args.img)
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor.to(device))
    
    # 后处理 + 可视化
    boxes, scores = postprocess(outputs, img, threshold=args.threshold)
    plot_results(img, boxes, scores, save_path=args.out)
