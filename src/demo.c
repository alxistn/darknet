#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"

#ifdef WIN32

#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"

#else

#include <sys/time.h>

#endif

#define FRAMES 3

#ifdef OPENCV

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"

#ifndef CV_VERSION_EPOCH

#include "opencv2/videoio/videoio_c.h"

#endif

image get_image_from_stream(CvCapture *cap);
void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
void show_image_cv_ipl(IplImage *disp, const char *name);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, IplImage** in_img, int use_webcam);

static char     **demo_names;
static image    **demo_alphabet;
static int      demo_classes;
static float    demo_thresh = 0;

static int      use_webcam = 0;
static float    fps = 0;

static network  net;

//static int      demo_index = 0;
int	*demo_index;

//static IplImage* ipl_images[FRAMES];

// Obscure

static float    *avg;
//static float    *predictions[FRAMES];
//static image    images[FRAMES];
static float    **probs;
static box      *boxes;

typedef struct	s_detection
{
  int		id;
  CvCapture	*cap;

  IplImage	*show_img;

  IplImage	*in_img;
  image		in;
  image		in_s;
  
  IplImage	*det_img;
  image		det;
  image		det_s;

  image		disp;

  IplImage	*ipl_images[FRAMES];

  image		images[FRAMES];
  float		*predictions[FRAMES];
}		t_detection;

void		*fetch_in_thread(void *ptr)
{
  t_detection	*d = (t_detection *)(ptr);
  
  d->in = get_image_from_stream_resize(d->cap, net.w, net.h, &d->in_img, use_webcam);

  if(!d->in.data){
    puts("Stream closed.");
    return NULL;
  }

  d->in_s = make_image(d->in.w, d->in.h, d->in.c);
  memcpy(d->in_s.data, d->in.data, d->in.h * d->in.w * d->in.c * sizeof(float));
  return NULL;
}

void		*detect_in_thread(void *ptr)
{
  t_detection	*d = (t_detection *)(ptr);
  float		nms = .4;
  float		*prediction = NULL;
  layer		l = net.layers[net.n-1];
  
  if (d->det_s.data == NULL)
    return NULL;

  prediction = network_predict(net, d->det_s.data);

  memcpy(d->predictions[demo_index[d->id]], prediction, l.outputs*sizeof(float));
  mean_arrays(d->predictions, FRAMES, l.outputs, avg);

  l.output = avg;

  free_image(d->det_s);

  if (l.type == DETECTION) {
    get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
  } else if (l.type == REGION){
    get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
  } else {
    error("Last layer must produce detections\n");
  }

  if (nms > 0) do_nms(boxes, probs, l.w * l.h * l.n, l.classes, nms);

  d->images[demo_index[d->id]] = d->det;
  d->det = d->images[(demo_index[d->id] + FRAMES / 2 + 1) % FRAMES];

  d->ipl_images[demo_index[d->id]] = d->det_img;
  d->det_img = d->ipl_images[(demo_index[d->id] + FRAMES / 2 + 1) % FRAMES];

  demo_index[d->id] = (demo_index[d->id] + 1) % FRAMES;

  draw_detections_cv(d->det_img, l.w * l.h * l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
  return 0;
}

double get_wall_time()
{
  struct timeval time;

  if (gettimeofday(&time,NULL)){
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void init(char *cfgfile, float thresh, char **names, int classes)
{
  demo_names = names;
  demo_alphabet = load_alphabet();;
  demo_classes = classes;
  demo_thresh = thresh;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char **filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
  int     delay = frame_skip;
  double  before = get_wall_time();
  double  after;
  t_detection	detection[42];

  int i = 0;
  int total = 0;

  while (filename[i]) {
    i++;
  }
  total = i;

  net = parse_network_cfg_custom(cfgfile, 1);

  layer l = net.layers[net.n - 1];
  int j;

  demo_index = calloc(total, sizeof(int));
  avg = (float *) calloc(l.outputs, sizeof(float));
  //for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
  //for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);
  boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
  probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
  for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

  while (i < total) {
    detection[i].id = i;
    detection[i].show_img = NULL;
    
    detection[i].disp.h = 0;
    detection[i].disp.w = 0;
    detection[i].disp.c = 0;
    detection[i].disp.data = NULL;
    
    detection[i].det_s.h = 0;
    detection[i].det_s.w = 0;
    detection[i].det_s.c = 0;
    detection[i].det_s.data = NULL;

    detection[i].det.h = 0;
    detection[i].det.w = 0;
    detection[i].det.c = 0;
    detection[i].det.data = NULL;

    for(j = 0; j < FRAMES; ++j) detection[i].predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) detection[i].images[j] = make_image(1,1,3);

    i++;
  }

  init(cfgfile, thresh, names, classes);

  printf("Demo\n");

  if (weightfile) {
    load_weights(&net, weightfile);
  }
  set_batch_network(&net, 1);
  srand(2222222);

  if (filename) {
    i = 0;

    while (i < total) {
      printf("video file: %s\n", filename[i]);
      if (!(detection[i].cap = cvCaptureFromFile(filename[i]))) error("Couldn't connect to webcam.\n");
      if (!prefix && !dont_show) {
	cvNamedWindow(filename[i], CV_WINDOW_NORMAL);
	cvMoveWindow(filename[i], 0, 0);
	cvResizeWindow(filename[i], 1352, 1013);
      }
      i++;
    }
  }

  pthread_t	fetch_thread[2];
  pthread_t	detect_thread[2];

  while (true) {
    i = 0;

    while (i < total) {
      if (pthread_create(&fetch_thread[i], NULL, fetch_in_thread, &detection[i])) error("Thread creation failed");
      if (pthread_create(&detect_thread[i], NULL, detect_in_thread, &detection[i])) error("Thread creation failed");

      if (!prefix && !dont_show && detection[i].show_img) {
	show_image_cv_ipl(detection[i].show_img, filename[i]);
	cvWaitKey(1);
      }
      
      cvReleaseImage(&detection[i].show_img);
      
      pthread_join(fetch_thread[i], NULL);
      pthread_join(detect_thread[i], NULL);
      
      if (delay == 0) {
	free_image(detection[i].disp);
	detection[i].disp = detection[i].det;
	detection[i].show_img = detection[i].det_img;
      }
      detection[i].det_img = detection[i].in_img;
      detection[i].det = detection[i].in;
      detection[i].det_s = detection[i].in_s;
      
      i++;
    }

    --delay;

    if(delay < 0){
      delay = frame_skip;
      after = get_wall_time();

      fps = 1./(after - before);
      before = after;
    }

  }

  printf("input video stream closed. \n");

}

#else

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
  fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}

#endif
