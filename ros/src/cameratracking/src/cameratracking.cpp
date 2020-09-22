#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
//#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include "cameratracking.hpp"
//#include <pcl_ros/point_cloud.h>

static inline tf::Vector3 createRot(const tf::Quaternion &src)
{
    return src.getAxis() * src.getAngle();
}

static inline Eigen::Quaterniond conv(const tf::Quaternion &src)
{
    return Eigen::Quaterniond(src.w(), src.x(), src.y(), src.z());
}

class D455CameraTracking
{
    tf::TransformListener listener;
    //ros::Subscriber sub_rgb;
    //ros::Subscriber sub_depth;
    //typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> RGBD;
    //message_filters::Synchronizer<RGBD> sync;
    message_filters::Subscriber<sensor_msgs::Image> sub_rgb;
    message_filters::Subscriber<sensor_msgs::Image> sub_depth;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync;
    ros::Subscriber sub_rgb_info;
    ros::Subscriber sub_depth_info;
    ros::Publisher pc_pub;
    //ros::Subscriber sub_pc;
    int skip_frame_cnt;
    bool rgbCameraInfoReady;
    bool depthCameraInfoReady;
    //sensor_msgs::PointCloud2 prevFrame;
    //sensor_msgs::PointCloud2 prevFrame;
    cv::Mat prevRGB;
    cv::Mat prevDepth;
    cv::Mat rgbCameraMat;
    cv::Mat rgbDistCoeff;
    cv::Mat depthCameraMat;
    cv::Mat depthDistCoeff;
    Pose3d prevPose;

    public:
    D455CameraTracking() : sync(sub_rgb, sub_depth, 2), skip_frame_cnt(10), rgbCameraInfoReady(false), depthCameraInfoReady(false)
    {
        const int N = 2;
        ros::NodeHandle n;
        sub_depth.subscribe(n, "camera/aligned_depth_to_color/image_raw", N);
        sub_rgb.subscribe(n, "camera/color/image_raw", N);
        //sub_depth      = n.subscribe("camera/aligned_depth_to_color/image_raw", N, &D455CameraTracking::depth_callback, this);
        //sub_rgb        = n.subscribe("camera/color/image_raw", N, &D455CameraTracking::rgb_callback, this);
        sub_depth_info = n.subscribe("/camera/aligned_depth_to_color/camera_info", N, &D455CameraTracking::depth_info_callback, this);
        sub_rgb_info   = n.subscribe("/camera/color/camera_info", N, &D455CameraTracking::rgb_info_callback, this);
        sync.registerCallback(boost::bind(&D455CameraTracking::rgbd_callback, this, _1, _2));

        //sub_pc = n.subscribe("/camera/depth/color/points", N, &D455CameraTracking::pc_callback, this);
        //pc_pub  = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/debug/pointcloud",N);

    }


    void setupCameraInfo(cv::Mat& cameraMat, cv::Mat distCoeff, const sensor_msgs::CameraInfo& camera)
    {
        cameraMat = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
        ROS_INFO("!!!!!!!!!!!! %f %f %f %f", camera.K.at(0), camera.K.at(4), camera.K.at(2), camera.K.at(5));
        cameraMat.at<double>(0, 0) = camera.K.at(0);//fx
        cameraMat.at<double>(1, 1) = camera.K.at(4);//fy
        cameraMat.at<double>(0, 2) = camera.K.at(2);//ppx
        cameraMat.at<double>(1, 2) = camera.K.at(5);//ppy
        cameraMat.at<double>(2, 2) = 1;

        distCoeff = cv::Mat::zeros(5, 1, cv::DataType<double>::type);;
        for (int i=0; i<5; i++) {
            distCoeff.at<double>(i) = camera.D.at(i);
        }
    }

    void convRGB(cv::Mat &mat, const sensor_msgs::ImageConstPtr &rgbmsg)
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(rgbmsg, sensor_msgs::image_encodings::BGR8);
        cv::undistort(cv_ptr->image, mat, rgbCameraMat, rgbDistCoeff);
    }

    void convDepth(cv::Mat &mat, const sensor_msgs::ImageConstPtr &depthmsg)
    {
        sensor_msgs::Image workaround;
        workaround.header = depthmsg->header;
        workaround.height = depthmsg->height;
        workaround.width = depthmsg->width;
        workaround.is_bigendian = depthmsg->is_bigendian;
        workaround.step = depthmsg->step;
        workaround.data = depthmsg->data;
        workaround.encoding = "mono16";
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(workaround, sensor_msgs::image_encodings::MONO16);
        cv::undistort(cv_ptr->image, mat, depthCameraMat, depthDistCoeff);
    }

    void rgb_info_callback(const sensor_msgs::CameraInfo &infomsg)
    {
        ROS_INFO("rgbinfo");
        sub_rgb_info.shutdown();
        setupCameraInfo(rgbCameraMat, rgbDistCoeff, infomsg);
        rgbCameraInfoReady = true;
    }

    void depth_info_callback(const sensor_msgs::CameraInfo &infomsg)
    {
        ROS_INFO("depthinfo");
        sub_depth_info.shutdown();
        setupCameraInfo(depthCameraMat, depthDistCoeff, infomsg);
        depthCameraInfoReady = true;
    }

    Pose3d getIMUPose(ros::Time stamp)
    {
        tf::StampedTransform transform;

        try{
            listener.lookupTransform("/realsense", "/base", stamp, transform);
        } catch (tf::TransformException &ex) {
            ROS_ERROR("%s",ex.what());
        }

        Pose3d pose;
        pose.q = conv(transform.getRotation());
        pose.p = prevPose.p;
        return pose;
    }

    void rgbd_callback(const sensor_msgs::ImageConstPtr rgbmsg, const sensor_msgs::ImageConstPtr depthmsg)
    {
        if (!rgbCameraInfoReady || !depthCameraInfoReady) {
            ROS_INFO("camera info is not ready.");
            return;
        }

        cv::Mat rgbMat;
        cv::Mat depthMat;

        convRGB(rgbMat, rgbmsg);
        convDepth(depthMat, depthmsg);

        if (skip_frame_cnt > 0) {
            skip_frame_cnt--;
            prevRGB = rgbMat;
            prevDepth = depthMat;
            return;
        }

        showImg("rgb", rgbMat);
        showImg("depth", depthMat, true);

        //Pose3d pose1_imu = getIMUPose(rgbmsg->header.stamp);
        Pose3d pose1_imu;
        Pose3d pose = track(prevRGB, rgbMat, prevDepth, depthMat, rgbCameraMat, rgbCameraMat, prevPose, pose1_imu);

        prevPose = pose;
        prevRGB = rgbMat;
        prevDepth = depthMat;
    }


/*
    void publishPointCloud()
    {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        double point[3];
        double pxl[2];
        for (int i = 0; i < prevRGB.rows; i++) {
            for (int j = 0; j < prevRGB.cols; j++) {
                pxl[0]=j;
                pxl[1]=i;

                deproject<double>(point, pxl, prevDepth.at<unsigned short>(i,j)/1000.0, rgbCameraMat.ptr<double>());

                if (point[2] <= 0) {
                    continue;
                }

                //project<double>(pxl, point, rgbCameraMat.ptr<double>());
                //ROS_ERROR("%d %d %f %f", i, j, pxl[1], pxl[0]);

                pcl::PointXYZRGB new_point;
                new_point.x = point[0];
                new_point.y = point[1];
                new_point.z = point[2];
                new_point.r = prevRGB.at<cv::Vec3b>(i,j)[2];
                new_point.g = prevRGB.at<cv::Vec3b>(i,j)[1];
                new_point.b = prevRGB.at<cv::Vec3b>(i,j)[0];
                cloud.points.push_back(new_point);
            }
        }
        auto msg = cloud.makeShared();
        msg->header.frame_id = "/base";
        pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
        pc_pub.publish(msg);
    }


*/
    //void pc_callback(const sensor_msgs::PointCloud2 &pcmsg)
    //{


    //}

    void exec()
    {
        ros::spin();
    }
};



int main(int argc, char **argv) {
    ros::init(argc, argv, "cameratracking");
    D455CameraTracking d455;
    d455.exec();
    return 0;
}