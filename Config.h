#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/core/core.hpp>

enum EXPRESS {HORIZONAL, VERTICAL};

class Config {
    protected:
        int width;
    public:
        virtual bool filter_region_by_shape(cv::Mat img)=0;
};

class HorizonalExpress: public Config {
    public:
        HorizonalExpress(int width);
        virtual bool filter_region_by_shape(cv::Mat img);
};

class VerticalExpress: public Config {
    public:
        VerticalExpress(int width);
        virtual bool filter_region_by_shape(cv::Mat img);

};

class ConfigFactory {
    
    public:
        Config* createConfig (enum EXPRESS type, int width) {
            if (type == HORIZONAL)
            {
                return new HorizonalExpress(width);
            } else if (type == VERTICAL)
            {
                return new VerticalExpress(width);
            } else {
                return NULL;
            }
        }
};

#endif