import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from config import TEST_DIR, DEVICE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from dataset import PascalVOCDataset, collate_fn
from backbone import build_model


def evaluate_detection_accuracy(model, dataloader, device, confidence_threshold=0.7, iou_threshold=0.75):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                scores = output['scores']
                boxes = output['boxes']
                labels = output['labels']

                keep = scores >= confidence_threshold
                pred_boxes = boxes[keep]
                pred_labels = labels[keep]

                gt_boxes = target['boxes']
                gt_labels = target['labels']

                matched = set()
                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_idx = -1
                    for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if i in matched:
                            continue
                        iou = torchvision.ops.box_iou(pb.unsqueeze(0), gb.unsqueeze(0)).item()
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    if best_iou >= iou_threshold and best_idx != -1 and pl.item() == gt_labels[best_idx].item():
                        total_tp += 1
                        matched.add(best_idx)
                    else:
                        total_fp += 1
                total_fn += len(gt_boxes) - len(matched)

    accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    return accuracy


if __name__ == '__main__':
    import torchvision
    dataset = PascalVOCDataset(TEST_DIR, transforms=T.Compose([T.ToTensor()]))
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    model = build_model(weights=None).to(DEVICE)
    model.load_state_dict(torch.load('/content/best_fasterrcnn.pth', map_location=DEVICE))
    acc = evaluate_detection_accuracy(model, loader, DEVICE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    print(f'✅ Detection Accuracy: {acc:.4f}')
