./darknet detector demo cfg/obj.data cfg/yolo-obj.cfg yolo.weights -thresh 0.80 'http://192.168.0.106:8080/?action=stream' 'http://192.168.0.109:8080/?action=stream' 
