import numpy as np
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2:
    def __init__(self, config):
        self.device = config["device"]

        checkpoint_path = config["checkpoint_path"]
        model_cfg_file = config["model_cfg"]
        self.video_mode = config["video_mode"]

        if self.video_mode == True:
            self.predictor = build_sam2_video_predictor(model_cfg_file, checkpoint_path, device=self.device)
        else:
            model = build_sam2(model_cfg_file, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(model)

    def segment_video(self, video_path, **kwargs):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = self.predictor.init_state(video_path=video_path)

            if "box" in kwargs:
                for i, box in enumerate(kwargs["box"]):
                    _ = self.predictor.add_new_points_or_box(
                            inference_state=state, 
                            frame_idx=0,
                            obj_id=i,
                            box=box
                    )
            if "points" in kwargs:
                for idx, point in enumerate(kwargs["points"]):
                    labels = np.zeros((len(kwargs["points"]),), dtype=np.int32)
                    labels[idx] = 1
                    _ = self.predictor.add_new_points_or_box(
                            inference_state=state, 
                            frame_idx=0,
                            obj_id=idx,
                            points=kwargs["points"],
                            labels=labels
                    )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                
        return video_segments

    def segment_image(self, image, **kwargs):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                **kwargs, 
                multimask_output=False
            )
        
        return masks, scores
    

            