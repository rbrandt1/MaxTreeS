/**
 * Max Tree Stereo Matching
 * manageImages.c
 *
 * Module of the program that implements all function needed to do I/O
 * operations on images.
 *
 */

#include "include/manageImages.hpp"

/**
 * Reads TIFF format image from the file which name is in fnm variable
 * and returns an array of pixels of the image.
 * @param fnm		Filename of the image to read
 * @param width		Width of the image read
 * @param height	Height of the image read
 * @return			Returns an array of pixels of the image read
 */
Pixel *ReadTIFF(const char *fnm, int *width, int *height){
	FIBITMAP *dib;
	unsigned long  bitsperpixel;
	Pixel *im;
	int x,y,i,imsize;

	dib = GenericLoader(fnm,0);
	if (dib == NULL) return NULL;

	bitsperpixel =  FreeImage_GetBPP(dib);
	*height = FreeImage_GetHeight(dib);
	*width = FreeImage_GetWidth(dib);
	imsize = (*width)*(*height);
	im = (Pixel*) calloc((size_t)imsize, sizeof(Pixel));
	assert(im!=NULL);
	i = 0;

	switch(bitsperpixel) {
		case 8:
			for(y = 0; y < *height; y++) {
				BYTE *bits = (BYTE *)FreeImage_GetScanLine(dib, y);
				for(x = 0; x < *width; x++,i++) {
					im[i] = bits[x];
				}
			}

			FreeImage_Unload(dib);
			return im;
		case 16:
			for(y = 0; y < *height; y++) {
				unsigned short *bits = (unsigned short *)FreeImage_GetScanLine(dib, y);
				for(x = 0; x < *width; x++,i++) {
					im[i] = bits[x];
				}
			}

			FreeImage_Unload(dib);
			return im;
		case 24:
			dib = FreeImage_ConvertTo8Bits(dib);
			for(y = 0; y < *height; y++) {
				BYTE *bits = (BYTE*)FreeImage_GetScanLine(dib, y);
				for(x = 0; x < *width; x++,i++) {
					im[i] = bits[x];
				}
			}

			FreeImage_Unload(dib);
			return im;
		default :
			FreeImage_Unload(dib);
			return NULL;
	}
}

/**
 * Writes pixels of an image width*height pointed by im variable into the
 * fname file.
 * @param	fname	Filename where to write the image
 * @param	im		Pointer to pixels of the image to write
 * @param	width	Width of the image to write
 * @param	height	Height of the image to write
 * @param	bpp		Bits Per Pixel of the image to write
 */
void WriteTIFF(const char *fname, Pixel *im, long width, long height, int bpp){
	FIBITMAP *outmap;
	long i,y,x;
	FREE_IMAGE_FORMAT fif;

	fif = FreeImage_GetFIFFromFilename(fname);
	if (bpp == 8){
		ubyte *imagebuf;
		RGBQUAD *pal;
		outmap = FreeImage_AllocateT(FIT_BITMAP, width, height, bpp, 0xFF, 0xFF,
																			0xFF);
		pal = FreeImage_GetPalette(outmap);
		for (i = 0; i < 256; i++) {
			pal[i].rgbRed = i;
			pal[i].rgbGreen = i;
			pal[i].rgbBlue = i;
		}
		i = 0;
		for (y = 0; y < height; y++){
			imagebuf = FreeImage_GetScanLine(outmap,y);
			for (x = 0; x < width; x++, i++)
				imagebuf[x] = im[i];
		}
	} else {
		uint16_t *imagebuf;
		outmap = FreeImage_AllocateT(FIT_UINT16, width, height, 16, 0xFFFF, 0xFFFF, 0xFFFF);
		i = 0;
		for (y = 0; y < height; y++){
			imagebuf = (uint16_t *)FreeImage_GetScanLine(outmap, y);
			for (x = 0; x < width; x++, i++)
				imagebuf[x] = im[i];
			for (x = 0; x < width; x++)
				printf("%d\n", imagebuf[x]);
		}
	}

	FreeImage_Save(fif, outmap, fname, 0);
	FreeImage_Unload(outmap);
}

/**
 * Generic image loader
 * @param	lpszPathName Pointer to the full file name
 * @param	flag Optional load flag constant
 * @return	Returns the loaded bitmap if successful, NULL otherwise
 */
FIBITMAP* GenericLoader(const char* lpszPathName, int flag) {
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	/*
	 * Extracts the file type from the signature of the file itself.
	 * If the file has no signature it tries to extract the format
	 * from the filename.
	 */
	fif = FreeImage_GetFileType(lpszPathName, 0);
	if(fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(lpszPathName);

	/**
	 * Checks if the library has reading capabilities for the format
	 * of the file. If it can, reads the file and returns the bitmap.
	 */
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) {
		FIBITMAP *dib = FreeImage_Load(fif, lpszPathName, flag);
		return dib;
	}
	return NULL;
}
