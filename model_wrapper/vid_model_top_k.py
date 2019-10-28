import torch
import torch.nn as nn

class I3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        # vid_t = vid_t * 2 - 1
        # vid_t = vid * 2 - 1
        vid_t.mul_(2).sub_(1)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            # vid = self.preprocess(vid)
            # out = self.model(vid)
            out = self.model(self.preprocess(vid))
        top_val, top_idx = torch.topk(out[0], k)
        return top_val, top_idx, out[1]

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)

class InceptionI3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        vid_t.mul_(2).sub_(1)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            out = self.model(self.preprocess(vid))
        logits = out.mean(2)
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)


class Lstm_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=vid.get_device())[None, :, None,
               None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=vid.get_device())[None, None, :, None,
              None]
        vid_t.sub_(mean).div_(std)
        # vid_t = vid_t.permute(0, 2, 1, 3, 4)
        # assert vid_t.size() == (1, 64, 3, 299, 299), vid_t.size()
        # vid_t = vid_t.squeeze(0)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            out = self.model(self.preprocess(vid))
        predict_value = torch.nn.functional.softmax(out,1)
        top_val, top_idx = torch.topk(predict_value, k)
        return top_val, top_idx, out

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)