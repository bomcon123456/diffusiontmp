from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, rootdir, filelist, transform):
        super().__init__()

        assert filelist.exists(), f"{filelist} does not exist."
        with open(filelist, "r") as f:
            filelist = f.readlines()
            filelist = list(map(lambda x: x.strip(), filelist))
            
        self.filelist = filelist
        self.rootdir = Path(rootdir)
        self.transform = transform
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, idx):
        filepath = self.rootdir / self.filelist[idx]
        image = Image.open(filepath.as_posix())
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        return {"input": image}

