### Basic Model

class SimpleConv(pl.LightningModule): # pl.LightningModule replaces nn.Module in pyTorch
    #def __init__(self, mode: ["classification", "regression"] = "classification"):
    def __init__(self, mode = "classification"):
        super().__init__()
        self.mode = mode
        # image size = 120*120, cropped from 576*576.
        
        self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=19, stride=7),
                    nn.Flatten(),
                )

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(3600, 500)
        self.fc2 = nn.Linear(500, 2)  # for classification
        self.fc3 = nn.Linear(500, 1)  # for regression


        self.stem = nn.Sequential(
            self.layer1, self.drop_out, self.fc1,
            )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, self.fc2)
        else:
            self.regression = nn.Sequential(self.stem, self.fc3)

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()