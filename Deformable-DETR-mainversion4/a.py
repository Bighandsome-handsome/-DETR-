import json
import matplotlib.pyplot as plt

def parse_full_log(log_file):
    epochs = []
    train_loss, test_loss = [], []
    train_ce, test_ce = [], []
    train_bbox, test_bbox = [], []
    train_giou, test_giou = [], []
    ap_values = []

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if "epoch" in data:
                epochs.append(data["epoch"])
                train_loss.append(data.get("train_loss", None))
                test_loss.append(data.get("test_loss", None))
                train_ce.append(data.get("train_loss_ce", None))
                test_ce.append(data.get("test_loss_ce", None))
                train_bbox.append(data.get("train_loss_bbox", None))
                test_bbox.append(data.get("test_loss_bbox", None))
                train_giou.append(data.get("train_loss_giou", None))
                test_giou.append(data.get("test_loss_giou", None))

                if "test_coco_eval_bbox" in data and len(data["test_coco_eval_bbox"]) > 0:
                    ap_values.append(data["test_coco_eval_bbox"][0])
                else:
                    ap_values.append(None)

    return (epochs, train_loss, test_loss,
            train_ce, test_ce,
            train_bbox, test_bbox,
            train_giou, test_giou,
            ap_values)

def plot_curves(epochs, train_loss, test_loss,
                train_ce, test_ce,
                train_bbox, test_bbox,
                train_giou, test_giou,
                ap_values, save_path):

    plt.figure(figsize=(12, 8))

    # 总 loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()

    # 分类 loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_ce, label="Train CE Loss")
    plt.plot(epochs, test_ce, label="Test CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss CE")
    plt.title("Classification Loss")
    plt.legend()

    # BBox loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_bbox, label="Train BBox Loss")
    plt.plot(epochs, test_bbox, label="Test BBox Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss BBox")
    plt.title("Bounding Box Loss")
    plt.legend()

    # GIoU loss + AP
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_giou, label="Train GIoU Loss")
    plt.plot(epochs, test_giou, label="Test GIoU Loss")
    plt.plot(epochs, ap_values, label="Test AP@[0.5:0.95]", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / AP")
    plt.title("GIoU Loss & AP")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    log_file = "/root/Deformable-DETR-mainversion4/exps/r50_deformable_detr/log.txt"
    (epochs, train_loss, test_loss,
     train_ce, test_ce,
     train_bbox, test_bbox,
     train_giou, test_giou,
     ap_values) = parse_full_log(log_file)

    # 绘制第1~50轮
    idx_50 = [i for i, e in enumerate(epochs) if e < 50]
    plot_curves([epochs[i] for i in idx_50],
                [train_loss[i] for i in idx_50],
                [test_loss[i] for i in idx_50],
                [train_ce[i] for i in idx_50],
                [test_ce[i] for i in idx_50],
                [train_bbox[i] for i in idx_50],
                [test_bbox[i] for i in idx_50],
                [train_giou[i] for i in idx_50],
                [test_giou[i] for i in idx_50],
                [ap_values[i] for i in idx_50],
                save_path="/root/Deformable-DETR-mainversion4/output/training_curves_1_50.png")

    # 绘制第1~100轮
    plot_curves(epochs, train_loss, test_loss,
                train_ce, test_ce,
                train_bbox, test_bbox,
                train_giou, test_giou,
                ap_values,
                save_path="/root/Deformable-DETR-mainversion4/output/training_curves_1_100.png")
