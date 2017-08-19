#pragma once

// Detection-related


// Neural Nets-related


#define real float
#define REAL_MAX FLT_MAX

#define EXPECTED_ALIGNMENT 8

#ifdef LINUX
	#define ALLOCATOR(T,SIZE) allocator<T>
#else
	#define ALLOCATOR(T,SIZE) aligned_allocator<T, SIZE>
#endif
#define CUSTOM_ALLOCATOR(T) ALLOCATOR(T,EXPECTED_ALIGNMENT)

// Saving some extra space for efficient intrinsic-based implementations to work correctly
#define RESERVED_SPACE 500

#define MAX_NUM_THREADS 1
#define BATCH_SIZE 1


