import time

import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.part0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.part1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.GELU(),
            nn.Linear(in_features=2048, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, x):
        x = self.part0(x)
        x = self.part1(x)
        return x


def cal_time1(model, x):
    with torch.inference_mode():
        time_list = []
        for _ in range(50):
            ts = time.perf_counter()
            ret = model(x)
            td = time.perf_counter()
            time_list.append(td - ts)

        print(f"avg time: {sum(time_list[5:]) / len(time_list[5:]):.5f}")


def cal_time2(model, x):
    device = x.device
    with torch.inference_mode():
        time_list = []
        for _ in range(50):
            torch.cuda.synchronize(device)
            ts = time.perf_counter()
            ret = model(x)
            torch.cuda.synchronize(device)
            td = time.perf_counter()
            time_list.append(td - ts)

        print(f"syn avg time: {sum(time_list[5:]) / len(time_list[5:]):.5f}")


def cal_time3(model, x):
    with torch.inference_mode():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        time_list = []
        for _ in range(50):
            start_event.record()
            ret = model(x)
            end_event.record()
            end_event.synchronize()
            time_list.append(start_event.elapsed_time(end_event) / 1000)

        print(f"event avg time: {sum(time_list[5:]) / len(time_list[5:]):.5f}")


def main():
    device = torch.device("cuda:0")
    model = CustomModel().eval().to(device)

    x = torch.randn(size=(32, 3, 224, 224), device=device)
    cal_time1(model, x)
    cal_time2(model, x)
    cal_time3(model, x)


if __name__ == '__main__':
    main()

