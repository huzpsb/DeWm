# DeWm
/djuːm/ DeWaterMark.  
[中文版](./README_CN.MD)  

Tired of images covered in watermarks? Look no further!  
This tool uses AI (StableDiffusion) to automatically remove watermarks from illustrations.  

**Disclaimer:** All sample images in this repository were generated by me.  
Removing watermarks from images does **not** mean you automatically gain authorization to use those images.  
More importantly, removing watermarks involves modifying the images, which is prohibited under many licenses. Please do not misuse the models provided in this repository!  

**Training Method:** No comment.  

## Usage:  
1. Use `dev.py` to process the original image and generate a mask.  
2. In Stable Diffusion (SD), select inpainting mode and upload the mask.  
That's it!  

### Tips for Better Results (not absolute; experiment on your own):  
1. Choose a model with a similar art style; it can significantly improve the quality of the final result.  
2. Avoid using blurred masks. Set mask blur to at most 1–2. Adjust the mathematical morphology operations provided in the function to refine the mask as needed.  
3. When in doubt, over-select rather than under-select. At worst, you can manually erase areas incorrectly selected by the model, which is still easier than manually blacking out watermarks.  
4. For semi-transparent watermarks, set masked areas to use the original image. Otherwise, use the filled option.  
5. I recommend using Euler a sampler with 20 steps.  
6. Keep the CFG scale low. If you can get by with 2, don’t use 3. The goal is to avoid breaking the image. Otherwise, the style may look overly "AI-generated." Ignore this if you prefer the AI aesthetic.  
7. Set denoising strength to 0.2, with a maximum of 0.25. Going higher will likely break the image because this process uses a fine mask.  
8. In extreme cases, first perform a "reverse AI" operation using [DeTox](https://github.com/huzpsb/DeTox).  

### Additional Uses:  
The inpainting process can also remove residual edges left by the watermark.  
Therefore, if you're planning to use ControlNet, you can first remove the watermark as described and then calculate edges from the watermark-free image.  

### TODO:  
If feasible, create a universal style inpainting model independent of (or based on) Stable Diffusion.  

## Examples:  
(Supports more types of transparent, image-based, and other watermarks; the following is for demonstration purposes.)  

Original Image:  
![](./sample.png)  
Generated Mask (you could manually erase false positives, but skipped here for demonstration):  
![](./mask.jpg)  
Stable Diffusion Inpainting:  
![](./ss.png)