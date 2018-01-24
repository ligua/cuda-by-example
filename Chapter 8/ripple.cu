#include "../common/book.h"
#include "../common/gpu_anim.h"

#define DIM 1024

__global__ void kernel( uchar4 *ptr, int ticks ) {
	// map from threadIdx/blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf( fx * fx + fy * fy );
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset].w = 255;
}

void generate_frame( uchar4 *pixels, void*, int ticks ) {
	dim3 grids( DIM / 16, DIM / 16 );
	dim3 threads( 16, 16 );
	kernel<<<grids, threads>>>( pixels, ticks );
}

struct GPUAnimBitmap {
	GLunit bufferObj;
	cudaGraphicsResource *resource;
	int width, height;
	void *dataBlock;
	void (*fAnim)(uchar4*, void*, int);
	void (*animExit)(void*);
	void (*clickDrag)(void*, int, int, int, int);
	int dragStartX, dragStartY;
}

GPUAnimBitmap( int w, int h, void* d ) {
	width = w;
	height = h;
	dataBlock = d;
	clickDrag = NULL;

	// first, find a CUDA device and set it to graphic interop
	cudaDeviceProp prop;
	int dev;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );

	cudaGLSetGLDevice( dev );

	int c = 1;
	char *foo = "name";
	glutInit( &c, &foo );

	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( width, height );
	glutCreateWindow( "bitmap" );

	glGenBuffers( 1, &bufferObj );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );

	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, NULL, GL_DYNAMIC_DRAW_ARB );

	HANDLE_ERROR( cudaGraphicsRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone ) );
}

// static method used for GLUT callbacks
static void idle_func( void ) {
	static int ticks = 1;
	GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
	uchar4* devPtr;
	size_t size;

	HANDLE_ERROR( cudaGraphicsMapResources( 1, &(bitmap -> resource), NULL ) );
	HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, bitmap -> resource ) );

	bitmap -> fAnim( devPtr, bitmap -> dataBlock, ticks++ );

	HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &(bitmap -> resource), NULL ) );

	glutPostRedisplay();
}

int main( void ) {
	GPUAnimBitmap bitmap( DIM, DIM, NULL );

	bitmap.anim_and_exit( (void (*)(uchar4*, void*, int))generate_frame, NULL );
}