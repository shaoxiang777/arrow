import os
import cv2 
import tools.Equirec2Perspec as E2P 

if __name__ == '__main__':
   image_path = 'data/image.jpg'
   output_image_path = 'data/image_perspec.jpg'
   equ = E2P.Equirectangular(image_path)    # Load equirectangular image

   #
   # FOV unit is degree 
   # theta is z-axis angle(right direction is positive, left direction is negative)
   # phi is y-axis angle(up direction positive, down direction negative)
   # height and width is output image dimension 

   img = equ.GetPerspective(60, 0, 0, 720, 1080) # Specify parameters(FOV, theta, phi, height, width)

   cv2.imwrite(output_image_path, img) 
   print(f"Image saved to: {output_image_path}")
   
   # python tools/equirec2perspec_image.py