import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from embed_time.splitter_static import DatasetSplitter
from embed_time.dataset_static import ZarrCellDataset, ZarrCellDataset_specific
from embed_time.dataloader_static import collate_wrapper
from datetime import datetime

time = datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_cell_data(dataset_image):
    sample = dataset_image
    images = [sample['original_image'], sample['cell_mask'], sample['nuclei_mask'], sample['cell_image'], sample['nuclei_image']]
    titles = ['Original', 'Cell Mask', 'Nuclei Mask', 'Cell Image', 'Nuclei Image']
    
    for i in range(2):  # Ensure cell and nuclei masks are 3D
        if images[i+1].ndim == 2:
            images[i+1] = images[i+1][None]

    num_channels = images[0].shape[0]
    fig, axes = plt.subplots(5, num_channels, figsize=(4*num_channels, 20))
    if num_channels == 1:
        axes = axes.reshape(-1, 1)
    
    for row, (image, title) in enumerate(zip(images, titles)):
        for channel in range(num_channels):
            im = axes[row, channel].imshow(image[channel], cmap='gray', vmin=-1 if row > 2 else None, vmax=1 if row > 2 else None)
            axes[row, channel].set_title(f'{title} - Channel {channel}')
            plt.colorbar(im, ax=axes[row, channel])
    
    for ax in axes.flatten():
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def print_cell_data_shapes(dataset_image):
    for key, value in dataset_image.items():
        print(f"{key}: {value.shape}")

def main(args):
    if args.generate_split and args.full:
        DatasetSplitter(args.parent_dir, args.output_dir, args.train_ratio, args.val_ratio, args.num_workers).generate_split()

    normalizations = v2.Compose([v2.CenterCrop(args.crop_size)])

    if args.full:
        dataset_class = ZarrCellDataset
        dataset_args = [args.parent_dir, args.csv_file, args.split, args.channels, args.mask, normalizations, None]
    else:
        dataset_class = ZarrCellDataset_specific
        dataset_args = [args.parent_dir, args.gene_name, args.barcode_name, args.channels, args.cell_cycle_stages, args.mask, normalizations, None]

    dataset = dataset_class(*dataset_args)

    print(f"The dataset contains {len(dataset)} images.")
    print(f"Dataset mean: {dataset.mean}")
    print(f"Dataset std: {dataset.std}")

    if args.plot_sample:
        plot_cell_data(dataset[args.sample_index])
        print_cell_data_shapes(dataset[args.sample_index])

    # save the dataset parameters and returned mean into a yaml file based on the datetime
    with open(f"/mnt/efs/dlmbl/G-et/yaml/dataset_info_{time}.yaml", "w") as file:
        file.write(f"Dataset mean: {dataset.mean}\n")
        file.write(f"Dataset std: {dataset.std}\n")
        file.write(f"Dataset length: {len(dataset)}\n")
        file.write(f"Dataset image shape: {dataset[0]['original_image'].shape}\n")
        file.write(f"Dataset nuclei shape: {dataset[0]['nuclei_image'].shape}\n")
        file.write(f"Dataset cell shape: {dataset[0]['cell_image'].shape}\n")
        file.write(f"Dataset cell mask shape: {dataset[0]['cell_mask'].shape}\n")
        file.write(f"Dataset nuclei mask shape: {dataset[0]['nuclei_mask'].shape}\n")
        file.write(f"Parent directory: {args.parent_dir}\n")
        if args.full:
            file.write(f"CSV file: {args.csv_file}\n")
            file.write(f"Split: {args.split}\n")
        else:
            file.write(f"Gene name: {args.gene_name}\n")
            file.write(f"Barcode name: {args.barcode_name}\n")
            file.write(f"Cell cycle stages: {args.cell_cycle_stages}\n")
        
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_wrapper(args.metadata_keys, args.images_keys)
    )

    # Print first batch info
    for batch in dataloader:
        print("First batch:")
        for key in args.metadata_keys + args.images_keys:
            if key in args.metadata_keys:
                print(f"{key}: {batch[key]}")
            else:
                print(f"{key} shape: {batch[key].shape}")
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE architecture for optical pooled screening data")
    parser.add_argument("--parent_dir", type=str, default="/mnt/efs/dlmbl/S-md/", help="Parent directory for dataset")
    parser.add_argument("--output_dir", type=str, default="/mnt/efs/dlmbl/G-et/csv/", help="Output file for dataset split")
    parser.add_argument("--generate_split", action="store_true", default=True, help="Generate dataset split")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio for dataset split")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio for dataset split")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of workers for dataset split")
    parser.add_argument("--full", action="store_true", default=True, help="Use full dataset (default: True)")
    parser.add_argument("--gene_name", type=str, default="AAAS", help="Gene name for specific dataset")
    parser.add_argument("--barcode_name", type=str, default="ATATGAGCACAATAACGAGC", help="Barcode name for specific dataset")
    parser.add_argument("--channels", nargs="+", type=int, default=[0, 1, 2, 3], help="Channels to use")
    parser.add_argument("--cell_cycle_stages", type=str, default="interphase", help="Cell cycle stages")
    parser.add_argument("--mask", type=str, default="masks", help="Mask type")
    parser.add_argument("--crop_size", type=int, default=100, help="Size for center crop")
    parser.add_argument("--csv_file", type=str, default="/home/S-md/embed_time/notebooks/splits/split_804.csv", help="CSV file for dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--plot_sample", action="store_true", help="Plot a sample from the dataset")
    parser.add_argument("--sample_index", type=int, default=10, help="Index of sample to plot")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dataloader")
    parser.add_argument("--metadata_keys", nargs="+", default=['gene', 'barcode', 'stage'], help="Metadata keys for collate function")
    parser.add_argument("--images_keys", nargs="+", default=['cell_image'], help="Image keys for collate function")

    args = parser.parse_args()
    main(args)
