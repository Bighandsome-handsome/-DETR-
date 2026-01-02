import json
import matplotlib.pyplot as plt

def parse_full_log(log_file):
    epochs = []
    train_loss, test_loss = [], []
    train_ce, test_ce = [], []
    train_bbox, test_bbox = [], []
    train_giou, test_giou = [], []
    ap_values = []
    card_error_train, card_error_test = [], []

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
                card_error_train.append(data.get("train_cardinality_error_unscaled", None))
                card_error_test.append(data.get("test_cardinality_error_unscaled", None))

                # COCO eval AP@[IoU=0.50:0.95]
                if "test_coco_eval_bbox" in data and len(data["test_coco_eval_bbox"]) > 0:
                    ap_values.append(data["test_coco_eval_bbox"][0])
                else:
                    ap_values.append(None)

    return (epochs, train_loss, test_loss,
            train_ce, test_ce,
            train_bbox, test_bbox,
            train_giou, test_giou,
            ap_values, card_error_train, card_error_test)


def plot_full_curves(log_file, save_path=None):
    (epochs, train_loss, test_loss,
     train_ce, test_ce,
     train_bbox, test_bbox,
     train_giou, test_giou,
     ap_values, card_error_train, card_error_test) = parse_full_log(log_file)

    plt.figure(figsize=(14, 10))

    # 总 loss
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()

    # 分类 loss
    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_ce, label="Train CE Loss")
    plt.plot(epochs, test_ce, label="Test CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss CE")
    plt.title("Classification Loss")
    plt.legend()

    # BBox loss
    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_bbox, label="Train BBox Loss")
    plt.plot(epochs, test_bbox, label="Test BBox Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss BBox")
    plt.title("Bounding Box Loss")
    plt.legend()

    # GIoU loss + AP
    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_giou, label="Train GIoU Loss")
    plt.plot(epochs, test_giou, label="Test GIoU Loss")
    plt.plot(epochs, ap_values, label="Test AP@[0.5:0.95]", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / AP")
    plt.title("GIoU Loss & AP")
    plt.legend()

    # Cardinality Error
    plt.subplot(3, 2, 5)
    plt.plot(epochs, card_error_train, label="Train Cardinality Error")
    plt.plot(epochs, card_error_test, label="Test Cardinality Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Cardinality Error")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    log_file = "/root/Deformable-DETR-mainversion3/exps/r50_deformable_detr/log.txt"  # 改成你的日志路径
    plot_full_curves(log_file, save_path="/root/Deformable-DETR-mainversion3/output/training_curves_full.png")
