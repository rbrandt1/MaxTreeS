/**
 * Max Tree Stereo Matching
 * manageImages.h
 *
 * Module of the program that implements all function needed to do I/O
 * operations on images.
 *
 * @author: A.Fortino
 */

#ifndef INCLUDE_MANAGEIMAGES_HPP_
#define INCLUDE_MANAGEIMAGES_HPP_

#include <FreeImage.h>
#include <assert.h>
#include "config.hpp"


/**
 * Reads TIFF format image from the file which name is in fnm variable
 * and returns an array of pixels of the image.
 * @param fnm		Filename of the image to read
 * @param width		Width of the image read
 * @param height	Height of the image read
 * @return			Returns an array of pixels of the image read
 */
Pixel* ReadTIFF(const char *fnm, int *width, int *height);

/**
 * Writes pixels of an image width*height pointed by im variable into the
 * fname file.
 * @param	fname	Filename where to write the image
 * @param	im		Pointer to pixels of the image to write
 * @param	width	Width of the image to write
 * @param	height	Height of the image to write
 * @param	bpp		Bits Per Pixel of the image to write
 */
void WriteTIFF(const char *fname, Pixel *im, long width, long height, int bpp);

/**
 * Generic image loader
 * @param	lpszPathName Pointer to the full file name
 * @param	flag Optional load flag constant
 * @return	Returns the loaded bitmap if successful, NULL otherwise
 */
FIBITMAP* GenericLoader(const char* lpszPathName, int flag);

#endif /* INCLUDE_MANAGEIMAGES_HPP_ */
