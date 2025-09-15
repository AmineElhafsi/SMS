import numpy as np
import torch
import torchvision.transforms.functional as F

from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import nms


class OWLv2:
    def __init__(self, config):
        self.device = config["device"]

        self.processor = Owlv2Processor.from_pretrained(config["model"])
        self.model = Owlv2ForObjectDetection.from_pretrained(config["model"]).to(self.device)
        self.score_threshold = config["score_threshold"]
        self.nms_threshold = config["nms_threshold"]
        self.texts = config["texts"]

    def detect(self, image, verbose=False):
        with torch.no_grad():
            if not isinstance(image, Image.Image):
                print("Converting to PIL Image")
                image = F.to_pil_image(image)
            
            inputs = self.processor(text=[self.texts], images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1]])
            
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)

            text = self.texts
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

            # nonmax suppression
            keep = nms(boxes, scores, iou_threshold=self.nms_threshold)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            box_list = []
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                if score >= self.score_threshold:
                    if verbose:
                        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
                    box_list.append(box)
                    detections.append(text[label])

            return detections, np.array(box_list), image