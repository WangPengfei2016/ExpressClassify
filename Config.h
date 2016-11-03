#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/core/core.hpp>


class Config {
    protected:
		cv::Rect bar;
    public:
		virtual cv::Mat cropy_region(cv::Mat region)=0;
};

class TempleteOne: public Config {
    public:
        TempleteOne(cv::Rect bar);
		virtual cv::Mat cropy_region(cv::Mat region);
};

class TempleteTwo: public Config {
    public:
        TempleteTwo(cv::Rect bar);
		virtual cv::Mat cropy_region(cv::Mat region);

};

class ConfigFactory {

    public:
        Config* createConfig (int type, cv::Rect bar) {
            if (type == 1)
            {
                return new TempleteOne(bar);
            } else if (type == 1)
            {
                return new TempleteTwo(bar);
            } else {
                return NULL;
            }
        }
};

#endif
