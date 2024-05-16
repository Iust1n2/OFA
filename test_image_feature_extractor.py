import torch.nn.functional as F
import torch 
import random

# Create a dummy batch of image patches with random data
# Batch size = 1, Channels = 3 (RGB), Height = 10, Width = 10
dummy_patches = torch.rand(1, 3, 10, 10)

def get_patch_images_info(self, patch_images, sample_patch_num, device):
        image_embed = self.embed_images(patch_images)
        h, w = image_embed.shape[-2:]
        image_num_patches = h * w
        image_padding_mask = patch_images.new_zeros(
            (patch_images.size(0), image_num_patches)).bool()
        image_position_idx = torch.arange(w).unsqueeze(0).expand(h, w) + \
            torch.arange(h).unsqueeze(1) * self.args.image_bucket_size + 1
        image_position_idx = image_position_idx.view(-1).to(device)
        image_position_ids = image_position_idx[None, :].expand(
            patch_images.size(0), image_num_patches)

        image_embed = image_embed.flatten(2).transpose(1, 2)
        if sample_patch_num is not None:
            patch_orders = [
                random.sample(range(image_num_patches), k=sample_patch_num)
                for _ in range(patch_images.size(0))
            ]
            patch_orders = torch.LongTensor(patch_orders).to(device)
            image_embed = image_embed.gather(1, patch_orders.unsqueeze(
                2).expand(-1, -1, image_embed.size(2)))
            image_num_patches = sample_patch_num
            image_padding_mask = image_padding_mask.gather(1, patch_orders)
            image_position_ids = image_position_ids.gather(1, patch_orders)
        orig_num_patches = (self.orig_patch_image_size // 16) ** 2
        orig_hw = self.orig_patch_image_size // 16
        if getattr(
            self.args,
            "interpolate_position",
                False) and image_num_patches > orig_num_patches:
            old_image_position_ids = torch.arange(orig_hw).unsqueeze(0).expand(
                orig_hw, orig_hw) + torch.arange(orig_hw).unsqueeze(1) * self.args.image_bucket_size + 1
            old_image_position_ids = old_image_position_ids.to(device)
            old_image_pos_embed = self.embed_image_positions(
                old_image_position_ids)
            old_image_pos_embed = old_image_pos_embed.reshape(
                1, orig_hw, orig_hw, -1).permute(0, 3, 1, 2)
            image_pos_embed = F.interpolate(
                old_image_pos_embed, size=(
                    h, w), mode='bilinear')
            image_pos_embed = image_pos_embed.permute(
                0, 2, 3, 1).reshape(
                1, image_num_patches, -1)
            image_pos_embed = image_pos_embed.expand(
                patch_images.size(0), -1, -1)
        else:
            image_pos_embed = self.embed_image_positions(image_position_ids)

        return image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed


# Set the device to 'cpu' or 'cuda' if you are using GPU
device = 'cpu'
sample_patch_num = 5  # Optional: specify a number of patches to sample

# Call the function
embeds, num_patches, padding_mask, position_ids = get_patch_images_info(dummy_patches, sample_patch_num, device)

# Print the outputs to observe changes
print("Embedded Patches Shape:", embeds.shape)
print("Number of Patches:", num_patches)
print("Padding Mask Shape:", padding_mask.shape)
print("Position IDs Shape:", position_ids.shape)
