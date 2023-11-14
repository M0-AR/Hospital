from PIL import Image

def convert_jpg_to_tiff(jpg_file, tiff_file):
    with Image.open(jpg_file) as img:
        img.save(tiff_file, format='TIFF')

# Example usage
convert_jpg_to_tiff('I90.jpg', 'test.tiff')
