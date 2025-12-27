# 这是 Emol 自主搭建的一个训练集

## 训练集的构成

### 数据集说明
1. 本数据集参考`HRSID`数据集，感谢他们团队发表的论文对本次训练集搭建的帮助（虽然我一个字也看不懂）。
2. 本训练集包含了中国上海-杭州（杭州湾）沿海和近海地区，主要裁剪了多个有代表性的港口和岛屿，以及近海部分海域船只密集的地区。
3. 本次训练集采用`COCO`格式进行标注。

### 图像来源
1. 我们的数据来源于[https://search.asf.alaska.edu/]网站下载的`Sentinel 1`卫星`SLC`版本的原始数据；
2. 在数据处理的过程中，我们使用`SNAP`软件提供的一些常规操作，具体可以参见我们的数据集处理过程部分（第1.2章）。
3. 在最后的噪声处理中，我们采用了包括但不限于以下的表达式，对我们的`VV`极化数据进行最后的增强：
   （1）简单线性增强；
   （2）对数变换，我们给出参考：`10 * log10(Sigma0_VV)`；
   （3）背景抑制，我们给出参考：`Sigma0_VV < 0.1 ? 0 : Sigma0_VV`；
   （4）幂次增强，我们给出参考：`pow(Sigma0_VV, 2)`。
   显然，我们对于靠近港口或者近海区域，采用(1)+(3)+(2)或者(1)+(3)+(4)的操作；
   对于那些深海地区的影像，我们采用(1)操作。



## 如何在自己的DMX上使用训练集

我们给出如何在`DETR`及其衍生模型上部署本训练集的方式。
1. 确认你的训练集和模型保持如下的文件结构，请不要修改Test01内部文件的名字。

    Your DMX's File /
    ├──Test01/
        ├── annotations/
        │   └── instances_val2017.json
        └── val2017/
            ├── tile_0002.png
            ├── tile_0005.png
            └── ...
    ├──dataset/...
    ├──data/...              `注意，不要放到data文件下，不要放到data文件下！`
    ├──other/...  

2. 为了保持良好的测试效果，我们建议修改`main.py`的`batch_size`参数如下，这样可以保证整个训练集得到测试。
   ```python
    def get_args_parser():           #  请在get_args_parser（传入参数）的函数中作修改
        ...
        parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
        parser.add_argument('--batch_size', default=1, type=int)              # batch_size 修改为 1
        parser.add_argument('--weight_decay', default=1e-4, type=float)
    ```
3. 为了避免有些模型不支持仅测试的功能，我们建议修改`main.py`的`def main()`函数的部分代码如下。
    ```python
    def main():
       ...
        if args.distributed:
            if not args.eval:
                if args.cache_mode:
                    sampler_train = samplers.NodeDistributedSampler(dataset_train)
                else:
                    sampler_train = samplers.DistributedSampler(dataset_train)
            # sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val,shuffle=False)
        else:
            if not args.eval:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if not args.eval:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)

            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                        pin_memory=True)
        else:
            data_loader_train = None

        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    pin_memory=True)
        ...
     ```
4. 关闭任何有关训练集的输出，例如输出训练集的长度等调试语句，具体地：
   ```python
   if args.resume:
        print(f"🔄 Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu',weights_only=False)
        # print(f"✅ 当前训练集样本数: {len(dataset_train)}")             
        # 请关闭上面这条print语句。
     ```
5. 下面，你可以使用命令行完成测试啦~
    ```bash
    python main.py \   
    --coco_path 'write your own path of this test-dataset(Test01)' \   ! 只写到Test01目录下的路径即可
    --dataset_file coco \  
    --eval   
    --resume 'write your own path of the reference model'

    ```
