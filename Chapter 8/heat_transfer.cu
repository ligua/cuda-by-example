void anim_gpu( uchar4* outputBitmap, DataBlock *d, int ticks ) {
	HANDLE_ERROR( cudaEventRecord( d -> start, 0 ) );
	dim3 blocks( DIM / 16, DIM / 16);
	dim3 threads( 16, 16 );

	// since tex is global and bound, we have to use a flag to
	// select which is in/out per iteration
	volatile bool dstOut = true;
	for (int i = 0; i < 90; i++) {
		float *in, *out;
		if (dstOut) {
			in = d -> dev_inSrc;
			out = d -> dev_outSrc;
		} else {
			out = d -> dev_inSrc;
			in = d -> dev_outSrc;
		}
		copy_const_kernel<<<blocks, threads>>>( in );
		blend_kernel<<<blocks, threads>>>( out, dstOut );
		dstOut = !dstOut;
	}
	float_to_color<<<blocks, threads>>>( outputBitmap, d -> dev_inSrc );
	
	HANDLE_ERROR( cudaEventRecord( d -> stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( d -> stop ) );
	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, d -> start, d -> stop ) );
	d -> totalTime += elapsedTime;
	++d -> frames;
	printf( "Average Time per frame:  %3.1f ms\n", d -> totalTime / d -> frames );
}

__global__ void float_to_color( unsigned char *optr, const float *outSrc ) {
	// convert floating-point value to 4-component color
	optr[offset * 4 + 0] = value( m1, m2, h + 120 );
	optr[offset * 4 + 1] = value( m1, m2, h );
	optr[offset * 4 + 2] = value( m1, m2, h - 120 );
	optr[offset * 4 + 3] = 255;
}

__global__ void float_to_color( uchar4 *optr, const float *outSrc ) {
	// convert floating-point value to 4-component color
	optr[offset].x = value( m1, m2, h + 120 );
	optr[offset].y = value( m1, m2, h );
	optr[offset].z = value( m1, m2, h - 120 );
	optr[offset].w = 255;
}

int main( void ) {
	DataBlock data;
	GPUAnimBitmap bitmap( DIM, DIM, &data );
	data.totalTime = 0;
	data.frames = 0;
	HANDLE_ERROR( cudaEventCreate( &data.start ) );
	HANDLE_ERROR( cudaEventCreate( &data.stop ) );

	int imageSize = bitmap.image_size();

	// assume float == 4 chars in size (i.e., rgba)
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc, imageSize ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc, imageSize ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc, imageSize ) );

	HANDLE_ERROR( cudaBindTexture( NULL, texConstSrc, data.dev_constSrc, imageSize ) );

	HANDLE_ERROR( cudaBindTexture( NULL, texIn, data.dev_inSrc, imageSize ) );

	HANDLE_ERROR( cudaBindTexture( NULL, texOut, data.dev_outSrc, imageSize ) );

	// initialize the constant data
	float *temp = (float*)malloc( imageSize );
	for (int i = 0; i < DIM * DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
			temp[i] = MAX_TEMP;
	}
	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;
	for (int y = 800; y < 900; y++) {
		for (int x = 400; x < 500; x++) {
			temp[x + y * DIM] = MIN_TEMP;
		}
	}
	HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice ) );

	// initialize the input data
	for (int y = 800; y < DIM; y++) {
		for (int x = 0; x < 200; x++) {
			temp[x + y * DIM] = MAX_TEMP;
		}
	}

	HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp, imageSize, cudaMemcpyHostToDevice ) );

	free(temp);

	bitmap.anim_and_exit( (void (*)(uchar4*, void*, int))anim_gpu, (void (*)(void*))anim_exit);
}