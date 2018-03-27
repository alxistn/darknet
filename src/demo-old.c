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
static int use_webcam = 0;
static float fps = 0;
static float demo_thresh = 0;

static float *avg;

void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
void show_image_cv_ipl(IplImage *disp, const char *name);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, IplImage** in_img, int use_webcam);


static CvCapture **cap = NULL;

static image in[2];
static image in_s[2];
static image det[2];
static image det_s[2];
static image disp[2] = {0};

IplImage* in_img[2];
IplImage* det_img[2];
IplImage* show_img[2];

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static IplImage* ipl_images[FRAMES];

static int flag_exit;

void *fetch_in_thread(void *ptr)
{
  //in = get_image_from_stream(cap);
  int	i = (int)(ptr);
  printf("YOLO: %d\n", (int)(ptr));

  //  while (cap[i] != NULL) {
  in[i] = get_image_from_stream_resize(cap[i], net.w, net.h, &(in_img[i]), use_webcam);
    if(!in[i].data){
      //error("Stream closed.");
      printf("Stream closed.\n");
      flag_exit = 1;
      return 0;
    }
    in_s[i] = make_image(in[i].w, in[i].h, in[i].c);
    memcpy(in_s[i].data, in[i].data, in[i].h*in[i].w*in[i].c*sizeof(float));
    //    i++;
    //  }
  //in_s = resize_image(in, net.w, net.h);
	
  return 0;
}
/*
void *detect_in_thread(void *ptr)
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
  }*/

double get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char **filenames, char **names, int classes, 
	  int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
  //skip = frame_skip;
  image		**alphabet = load_alphabet();
  int		delay = frame_skip;
  demo_names = names;
  demo_alphabet = alphabet;
  demo_classes = classes;
  demo_thresh = thresh;
  printf("Demo\n");
  net = parse_network_cfg_custom(cfgfile, 1);
  if(weightfile){
    load_weights(&net, weightfile);
  }
  set_batch_network(&net, 1);
  
  srand(2222222);
  
  int total = 0;
  
  if(filenames) {
    int i = 0;

    while (filenames[i] != NULL) {
	printf("video file: %s\n", filenames[i++]);
    }
    total = i;
    
    if ((cap = malloc(sizeof(*cap) * total)) != NULL) {
      i = 0;
      
      while (filenames[i] != NULL) {
	if (!(cap[i] = cvCaptureFromFile(filenames[i]))) {
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
      cap[i] = NULL;
      //      show_img[i] = NULL;
    }
    
  }else{
    /*       printf("Webcam index: %d\n", cam_index);
#ifdef CV_VERSION_EPOCH	// OpenCV 2.x
    cap = cvCaptureFromCAM(cam_index);
#else					// OpenCV 3.x
    use_webcam = 1;
    cap = get_capture_webcam(cam_index);
#endif
    */
  }

  layer l = net.layers[net.n-1];
  int j;
  
  avg = (float *) calloc(l.outputs, sizeof(float));
  for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
  for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);
  
  boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
  probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
  for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
  
  flag_exit = 0;
  
  pthread_t fetch_thread_0;
  pthread_t fetch_thread_1;
  pthread_t detect_thread;

  int count = 0;
  
  double before = get_wall_time();
  int i = 0;
  while(1){
    ++count;
    if(1){
      // create the threads
      if(pthread_create(&fetch_thread_0, 0, fetch_in_thread, 0)) error("Thread creation failed");
      if(pthread_create(&fetch_thread_1, 1, fetch_in_thread, 1)) error("Thread creation failed");
      //      if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
      
      if(!prefix){
	if (!dont_show) {
	  i = 0;
	  while (filenames[i] != NULL) {
	    show_image_cv_ipl(show_img[i], filenames[i]);
	    i++;
	  }
	  // important
	  int c = cvWaitKey(1);
	  /*	  if (c == 10) {
	    if (frame_skip == 0) frame_skip = 60;
	    else if (frame_skip == 4) frame_skip = 0;
	    else if (frame_skip == 60) frame_skip = 4;
	    else frame_skip = 0;
	    }*/
	}
      } else{
	/*	char buff[256];
	//	sprintf(buff, "%s_%08d", prefix, count);
	save_image(disp, buff);*/
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
	cvReleaseImage(&show_img[i++]);
      }
	
      pthread_join(fetch_thread_0, 0);
      pthread_join(fetch_thread_1, 1);
      //      pthread_join(detect_thread, 0);
      
      if (flag_exit == 1) break;
      
      if(delay == 0){
	i = 0;
	while (i < total) {	
	  free_image(disp[i]);
	  disp[i]  = det[i];
	  show_img[i] = det_img[i];
	  i++;
	}
      }
      i = 0;
      while (i < total) {
	det_img[i] = in_img[i];
	det[i]   = in[i];
	det_s[i] = in_s[i];
	i++;
      }
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
  if (output_video_writer) {
    cvReleaseVideoWriter(&output_video_writer);
    printf("output_video_writer closed. \n");
  }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename, int http_stream_port, int dont_show)
{
  fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

