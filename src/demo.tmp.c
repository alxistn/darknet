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

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
//static CvCapture **cap = NULL;
static int use_webcam = 0;
static float fps = 0;
static float demo_thresh = 0;

static float *avg;

void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
void show_image_cv_ipl(IplImage *disp, const char *name);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, IplImage** in_img, int use_webcam);

image disp = {0};
typedef struct		s_detection
{
  int		id;
  IplImage*		in_img;
  IplImage*		det_img;
  IplImage*		show_img;

  CvCapture	*cap;

  image		in;
  image		in_s;
  image		det;
  image		det_s;
  image		disp;
  
  image		images[FRAMES];
  IplImage*	ipl_images[FRAMES];
  float		*predictions[FRAMES];
  
  int		demo_index;
}			t_detection;

//disp[0] = {0};
//disp[1] = {0};
/*
IplImage* in_img;
IplImage* det_img;
IplImage* show_img;

static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};

static image images[FRAMES];
static IplImage* ipl_images[FRAMES];
static float *predictions[FRAMES];

static int demo_index = 0;
*/

static int flag_exit;

void	*fetch_in_thread(void *ptr)
{
  //in = get_image_from_stream(cap);
  //  int	i = (int)(ptr);
  t_detection	*detection = (t_detection *)(ptr);

  printf("FETCH-%d-1\n", detection->id);

  detection->in = get_image_from_stream_resize(detection->cap, net.w, net.h, &(detection->in_img), use_webcam);
  printf("FETCH-%d-2\n", detection->id);
  if(!detection->in.data){
    printf("FETCH-%d-3\n", detection->id);
    printf("Stream closed.\n");
    flag_exit = 1;
    return 0;
  }

  image in = detection->in;
  printf("FETCH-%d-4\n", detection->id);
  detection->in_s = make_image(in.w, in.h, in.c);
  printf("FETCH-%d-5\n", detection->id);
  memcpy(detection->in_s.data, in.data, in.h * in.w * in.c * sizeof(float));
  printf("FETCH-%d-6\n", detection->id);
  return 0;
}
/*
void	*fetch_in_thread(void *ptr)
{
  //in = get_image_from_stream(cap);
  int	i = 0;
  printf("YOLO: %d\n", (int)(ptr));

  in = get_image_from_stream_resize(cap[0], net.w, net.h, &in_img, use_webcam);
  if(!in.data){
    printf("Stream closed.\n");
    flag_exit = 1;
    return 0;
  }
  in_s = make_image(in.w, in.h, in.c);
  memcpy(in_s.data, in.data, in.h*in.w*in.c*sizeof(float));
  return 0;
}
*/
void *detect_in_thread(void *ptr)
{
  //  int i = (int)(ptr);
  t_detection	*detection = (t_detection *)(ptr);
  float nms = .4;

  layer l = net.layers[net.n-1];
  float *X = detection->det_s.data;
  float *prediction = network_predict(net, X);

  printf("DETECT-%d\n", detection->id);

  memcpy(detection->predictions[detection->demo_index], prediction, l.outputs*sizeof(float));
  mean_arrays(detection->predictions, FRAMES, l.outputs, avg);
  l.output = avg;

  free_image(detection->det_s);
  if(l.type == DETECTION){
    get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
  } else if (l.type == REGION){
    get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
  } else {
    error("Last layer must produce detections\n");
  }
  if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
  /*  printf("\033[2J");
  printf("\033[1;1H");
  printf("\nFPS:%.1f\n",fps);
  printf("Objects:\n\n");
  */
  
  detection->images[detection->demo_index] = detection->det;
  detection->det = detection->images[(detection->demo_index + FRAMES/2 + 1)%FRAMES];
  detection->ipl_images[detection->demo_index] = detection->det_img;
  detection->det_img = detection->ipl_images[(detection->demo_index + FRAMES / 2 + 1) % FRAMES];
  detection->demo_index = (detection->demo_index + 1)%FRAMES;
	    
  //draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
  draw_detections_cv(detection->det_img, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

  return 0;
}
/*void *detect_in_thread(void *ptr)
{
   float nms = .4;

  layer l = net.layers[net.n-1];
  float *X = det_s.data;
  float *prediction = network_predict(net, X);

  memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
  mean_arrays(predictions, FRAMES, l.outputs, avg);
  l.output = avg;

  free_image(det_s);
  if(l.type == DETECTION){
    get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
  } else if (l.type == REGION){
    get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
  } else {
    error("Last layer must produce detections\n");
  }
  if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
  /*  printf("\033[2J");
  printf("\033[1;1H");
  printf("\nFPS:%.1f\n",fps);
  printf("Objects:\n\n");
  
  
  images[demo_index] = det;
  det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
  ipl_images[demo_index] = det_img;
  det_img = ipl_images[(demo_index + FRAMES / 2 + 1) % FRAMES];
  demo_index = (demo_index + 1)%FRAMES;
	    
  //draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
  draw_detections_cv(det_img, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

  return 0;
}
*/
double get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void	demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char **filenames, char **names, int classes, int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
  image	**alphabet = load_alphabet();
  t_detection	*detection;
  int	delay = frame_skip;
  int	total = 0;
  int	i = 0;

  printf("Demo\n");

  demo_names = names;
  demo_alphabet = alphabet;
  demo_classes = classes;
  demo_thresh = thresh;
  net = parse_network_cfg_custom(cfgfile, 1);

  layer l = net.layers[net.n-1];
  int j;
  
  avg = (float *) calloc(l.outputs, sizeof(float));  
  boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
  probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
  for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

  if (weightfile) {
    load_weights(&net, weightfile);
  }
  set_batch_network(&net, 1);
  srand(2222222);
  
  if(filenames) {
    i = 0;

    while (filenames[i] != NULL) {
	printf("video file: %s\n", filenames[i++]);
    }
    total = i;

    if ((detection = malloc(sizeof(*detection) * total)) == NULL) {
      return error("Error: malloc failed\n");
    }
    i = 0;
    
    while (filenames[i] != NULL) {
      j = 0;

      detection[i].id = i;
      detection[i].disp.h = 0;
      detection[i].disp.w = 0;
      detection[i].disp.c = 0;
      detection[i].disp.data = NULL;
      detection[i].demo_index = 0;
      detection[i].show_img = NULL;
      while (j < FRAMES) {
	detection[i].predictions[j] = (float *) calloc(l.outputs, sizeof(float));
	detection[i].images[j] = make_image(1,1,3);
	j++;
      }
      
      if (!(detection[i].cap = cvCaptureFromFile(filenames[i]))) {
	printf("%s: could not connect to stream\n", filenames[i]);
	error("Couldn't connect to webcam.\n");
      } else {
	if(!prefix && !dont_show){
	  cvNamedWindow(filenames[i], CV_WINDOW_NORMAL); 
	  cvMoveWindow(filenames[i], 0, 0);
	  cvResizeWindow(filenames[i], 1352, 1013);
	}
      }
      i++;
    }
  } 


  flag_exit = 0;
  
  pthread_t fetch_thread_0;
  pthread_t fetch_thread_1;
  pthread_t detect_thread_0;
  pthread_t detect_thread_1;
  /*
  typedef struct	s_stream
  {
    pthread_t		t_id;
    
    }		t_stream;*/

  int count = 0;
  
  double before = get_wall_time();
  i = 0;
  while(1){
    ++count;
    if(1){
      // create the threads
      if(pthread_create(&fetch_thread_0, NULL, fetch_in_thread, &(detection[0]))) error("Thread creation failed");
      if(pthread_create(&fetch_thread_1, NULL, fetch_in_thread, &(detection[1]))) error("Thread creation failed");
      //if(pthread_create(&detect_thread_0, NULL, detect_in_thread, &(detection[0]))) error("Thread creation failed");
      //if(pthread_create(&detect_thread_1, NULL, detect_in_thread, &(detection[1]))) error("Thread creation failed");
      
      if(!prefix){
	if (!dont_show) {
	  i = 0;

	  while (filenames[i] != NULL && detection[i].show_img != NULL) {
	    puts("DEBUG-x1");
	    show_image_cv_ipl(detection[i].show_img, filenames[i]);
	    puts("DEBUG-x2");
	    i++;
	  }
	  cvWaitKey(1);
	}
      }
      
      // if you run it with param -http_port 8090  then open URL in your web-browser: http://localhost:8090
      /*
      if (http_stream_port > 0 && show_img) {
	puts("DEBUG-5");
	//int port = 8090;
	int port = http_stream_port;
	int timeout = 200;
	int jpeg_quality = 30;	// 1 - 100
	send_mjpeg(show_img, port, timeout, jpeg_quality);
	}*/
      
      // save video file
      /*      if (output_video_writer && show_img) {
	puts("DEBUG-6");
	cvWriteFrame(output_video_writer, show_img);
	printf("\n cvWriteFrame \n");
      }
      */

      i = 0;

      while (i < total) {
	puts("DEBUG-x2");
	cvReleaseImage(&(detection[i++].show_img));
	puts("DEBUG-x2");
      }
      
      pthread_join(fetch_thread_0, NULL);
      pthread_join(fetch_thread_1, NULL);
      //pthread_join(detect_thread_0, NULL);
      //pthread_join(detect_thread_1, NULL);
    
      if (flag_exit == 1) break;
      
      if(delay == 0){
	i = 0;
	
	while  (i < total) {
	  free_image(detection[i].disp);
	  detection[i].disp  = detection[i].det;
	  detection[i].show_img = detection[i].det_img;
	  i++;
	  puts("DEBUG-x3");
	}
      }
      i = 0;
      
      while  (i < total) {
	detection[i].det_img = detection[i].in_img;
	detection[i].det = detection[i].in;
	detection[i].det_s = detection[i].in_s;
	i++;
	puts("DEBUG-x4");
      }
      /*
      if(delay == 0){
	free_image(disp);
	disp  = det;
	show_img = det_img;
      }
      det_img = in_img;
      det   = in;
      det_s = in_s;
      */
    }else {
      /*
      fetch_in_thread(0);
      det_img = in_img;
      det   = in;
      det_s = in_s;
      detect_in_thread(0);
      if(delay == 0) {
	free_image(disp);
	disp = det;
      }
      if (!dont_show) {
	i = 0;
	while (filenames[i] != NULL) {
	  show_image(disp, filenames[i++]);
	}
	cvWaitKey(1);
      }
      */
    }
    --delay;
    if(delay < 0){
      delay = frame_skip;
      
      double after = get_wall_time();
      float curr = 1./(after - before);
      fps = curr;
      before = after;
    }
  }
  printf("input video stream closed. \n");
  /*  if (output_video_writer) {
    cvReleaseVideoWriter(&output_video_writer);
    printf("output_video_writer closed. \n");
    }*/
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
  fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

