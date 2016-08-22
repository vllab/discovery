#include "mex.h"
#include "math.h"
#include <algorithm>
#include <vector>
using namespace std;

// trivial array class encapsulating pointer arrays
template <class T> class Array
{
public:
  Array() { _h=_w=0; _x=0; _free=0; }
  virtual ~Array() { clear(); }
  void clear() { if(_free) delete [] _x; _h=_w=0; _x=0; _free=0; }
  void init(int h, int w) { clear(); _h=h; _w=w; _x=new T[h*w](); _free=1; }
  T& val(size_t r, size_t c) { return _x[c*_h+r]; }
  int _h, _w; T *_x; bool _free;
};

// convenient typedefs
typedef vector<float> vectorf;
typedef vector<int> vectori;
typedef Array<float> arrayf;
typedef Array<int> arrayi;

// bounding box data structures and routines
typedef struct { int c, r, w, h; } Box;  
typedef vector<Box> Boxes;
float boxesIntersection( Box &a, Box &b );
 
// main class
class fullOverlap
{
public:
	// parameters
	int _num;
	// main function
	void processing( Boxes &boxes, arrayf &input, arrayf &output);  
};

////////////////////////////////////////////////////////////////////////////////
void fullOverlap::processing( Boxes &boxes, arrayf &input, arrayf &output ) 
{
	// read proposals
	boxes.resize(0);
	for( int i=0; i<input._h; i++ )
	{ Box b; b.c=input.val(i,0); b.r=input.val(i,1); b.w=input.val(i,2); b.h=input.val(i,3); boxes.push_back(b);	}
	// calculatiing the overlap of each box pair
	for( int i=0; i<_num; i++ ) 
		{ for( int j=i+1; j<_num; j++ ) { output.val(i,j) = boxesIntersection( boxes[i], boxes[j] ); output.val(j,i) = output.val(i,j);} }
	for( int i=0; i<_num; i++ ) { output.val(i,i) = 1; }
}

float boxesIntersection( Box &a, Box &b ) 
{
	float areaA, areaB, areaAB;
	int r0, r1, c0, c1, r1i, c1i, r1j, c1j;
	r1i=a.r+a.h; c1i=a.c+a.w; if( a.r>=r1i || a.c>=c1i ) return 0;
	r1j=b.r+b.h; c1j=b.c+b.w; if( a.r>=r1j || a.c>=c1j ) return 0;
	areaA  = (float) a.w*a.h; r0=max(a.r,b.r); r1=min(r1i,r1j);
	areaB  = (float) b.w*b.h; c0=max(a.c,b.c); c1=min(c1i,c1j);
	areaAB = (float) max(0,r1-r0)*max(0,c1-c0);
	return areaAB;
}

////////////////////////////////////////////////////////////////////////////////
// Matlab entry point: 
void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] )
{
	// get inputs
	arrayf input; input._x = (float*) mxGetData(pr[0]); // input bbox
	int hh = (int) mxGetM(pr[0]); input._h = hh;        // # proposals
	int ww = (int) mxGetN(pr[0]); input._w = ww;        // # sides

	// create output 
	arrayf output;
	{   
		pl[0] = mxCreateNumericMatrix(hh, hh, mxSINGLE_CLASS, mxREAL);
		output._x = (float *) mxGetData(pl[0]); 
		output._h = hh; 
		output._w = hh;
	}	
 
	// setup and run fullOverlap
	fullOverlap PPS; 
	Boxes boxes;
	PPS._num = hh;
	PPS.processing( boxes, input, output );             // function
}
