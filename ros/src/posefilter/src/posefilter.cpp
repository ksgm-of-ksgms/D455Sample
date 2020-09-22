#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <Eigen/Core>
#include <math.h>


static const float GRAVITY_ACC = 9.80665f;

static inline tf::Quaternion createQuaternion(const tf::Vector3 &src)
{
    if (src.length() < 0.0000000000000001) {
        return tf::Quaternion(tf::Vector3(1, 0, 0), 0);
    }
    return tf::Quaternion(src.normalized(), src.length());
}

static inline tf::Vector3 createRot(const tf::Quaternion &src)
{
    return src.getAxis() * src.getAngle();
}

static inline tf::Vector3 conv(const geometry_msgs::Vector3& src)
{
    tf::Vector3 ret;
    tf::vector3MsgToTF(src, ret);
    return ret;
}

static inline geometry_msgs::Vector3 conv(const tf::Vector3& src)
{
    geometry_msgs::Vector3 ret;
    tf::vector3TFToMsg(src, ret);
    return ret;
}

static inline geometry_msgs::Quaternion conv(const tf::Quaternion& src)
{
    geometry_msgs::Quaternion ret;
    tf::quaternionTFToMsg(src, ret);
    return ret;
}

static inline void info(const char* title, const tf::Vector3& x)
{
    ROS_INFO("%s: %f %f %f", title, x.getX(), x.getY(), x.getZ());
    //double r,p,y;
    //tf::Matrix3x3(createQuaternion(x)).getRPY(r,p,y);
    //ROS_INFO("%s2: %f %f %f", title, (float) 180 / M_PI * r, (float)180 / M_PI * p, (float)180 / M_PI * y);
}

static inline void info(const char* title, const tf::Quaternion& x)
{
    double r,p,y;
    tf::Matrix3x3(x).getRPY(r,p,y);
    //ROS_INFO("%s2: %f %f %f", title, (float) 180 / M_PI * r, (float)180 / M_PI * p, (float)180 / M_PI * y);
}

static inline void tf_quat_to_rpy(double& roll, double& pitch, double& yaw, tf::Quaternion quat){
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
}

static inline tf::Quaternion rpy_to_tf_quat(double roll, double pitch, double yaw){
    return tf::createQuaternionFromRPY(roll, pitch, yaw);
}

static inline double length2(double x, double y)
{
    return sqrt(x*x + y*y);
}

static inline void EstimateRP(double& rall, double& pitch, const tf::Vector3& a)
{
    pitch = atan2(-a.getX(), length2(a.getY(), a.getZ()));
    rall  = atan2(a.getY(), a.getZ());
}

static inline tf::Quaternion EstimatePoseByAcc(const double& yaw, const tf::Vector3& a)
{
    double r,p;
    EstimateRP(r, p, a);
    return tf::createQuaternionFromRPY(r, p, yaw);
}


class PoseFilter
{
public:
    tf::Vector3 bias_omega;
    tf::Vector3 scale_acc;
    tf::Vector3 bias_acc;
    tf::Quaternion accPose;
    tf::Quaternion gyroPose;
    tf::Quaternion pose;
    tf::Vector3 position;
    tf::Vector3 velocity;
    tf::Vector3 acc_cached;
    double alpha;
    double tprev;
    int skipFrame;
public:
    enum {
        COMP_FILTER = 0,
        GYRO = 1,
        ACC = 2,

    };
    PoseFilter(double alpha = 0.95, int skipFrame = 100)
        : pose(tf::Vector3(0, 0, 1), 0), alpha(alpha), velocity(0, 0, 0), position(0, 0, 0), skipFrame(skipFrame)
    {
        scale_acc = tf::Vector3(1, 1, 1);
        bias_acc = tf::Vector3(0, 0, 0);
    }

    tf::Vector3 GetPosition()
    {
        return position;
    }

    tf::Quaternion GetPose(int type = 0)
    {
        if (type == ACC) {
            return accPose;
        } else if (type == GYRO) {
            return gyroPose;
        } else {
            return pose;
        }
    }

    void SetData(const double t, const tf::Vector3 &acc, const tf::Vector3 &omega, bool reset = false)
    {
        float dt;

        if (reset) {
            dt = 0;
        } else {
            dt = (t - tprev);
        }
        tprev = t;

        if (skipFrame > 0) {
            skipFrame--;
            return;
        }

        double r,p,y = 0;
        tf_quat_to_rpy(r, p, y, pose);
        //tf::Vector3 acc1 = acc; // * scale_acc;
        //info("acc_org", acc);
        tf::Vector3 acc1 = (acc + bias_acc) * scale_acc;
        accPose = EstimatePoseByAcc(y, acc1);

        tf::Vector3 gyro = (omega - bias_omega) * dt;
        gyroPose = pose * tf::Quaternion(gyro.getX()/2, gyro.getY()/2, gyro.getZ()/2, 1);

        //pose = gyroPose.slerp(accPose, alpha);
        pose = accPose.slerp(gyroPose, alpha);
        //tf::Quaternion acc1 = pose * tf::Quaternion(, 0) * pose.inverse();
        tf::Vector3 cor_acc = tf::Matrix3x3(pose) * acc1 - tf::Vector3(0, 0, GRAVITY_ACC); // ここで重力定数保証
        //info("acc_cor", acc1);
        acc_cached = cor_acc;
        position = position + dt * velocity;
        velocity = velocity + cor_acc * dt;
        //info("acc : ", acc_cached);
        //info("vel : ", velocity);
        //info("pos : ", position);
    }
};

class D455PoseFilter
{
    tf::TransformBroadcaster tf_broadcaster;
    //ros::Publisher sst;
    ros::Subscriber sub;
    ros::Subscriber sub1;
    bool resetIntegration;
    PoseFilter posefilter;
public:
    D455PoseFilter()
        : resetIntegration(true), posefilter(PoseFilter())
    {
        posefilter.bias_omega = tf::Vector3(-0.003035, 0.000007, -0.000134);
        tf::Vector3 max_acc = tf::Vector3(9.72135297916667, 9.90708857142857 ,  10.1711068571429 );
        tf::Vector3 min_acc = tf::Vector3(-9.4777658      , -9.21534608333333,  -9.01180233333333);
        tf::Vector3 scale = ((max_acc - min_acc) / 2.0f);
        posefilter.scale_acc = GRAVITY_ACC * tf::Vector3(1.0f / scale.x(), 1.0f / scale.y(), 1.0f / scale.z());
        posefilter.bias_acc = - (max_acc + min_acc) / 2.0f;

        ros::NodeHandle n;
        sub = n.subscribe("camera/imu", 100, &D455PoseFilter::callback, this);
        sub1 = n.subscribe("acccalib", 1, &D455PoseFilter::handleAccCalib, this);
    }

    void handleAccCalib(const geometry_msgs::Vector3 &msg)
    {
        float x = posefilter.acc_cached.getX();
        float y = posefilter.acc_cached.getY();
        float z = posefilter.acc_cached.getZ();
        if ((fabs(x) < 0.1) && (fabs(y) < 0.1)) {
            ROS_INFO("SAVE1 %f %f %f (%f)", x, y, z, sqrt(x*x + y*y + z*z));
        }
        else if (fabs(x) < 0.1 && fabs(z) < 0.1) {
            ROS_INFO("SAVE2 %f %f %f (%f)", x, y, z, sqrt(x*x + y*y + z*z));
        }
        else if (fabs(y) < 0.1 && fabs(z) < 0.1) {
            ROS_INFO("SAVE3 %f %f %f (%f)", x, y, z, sqrt(x*x + y*y + z*z));
        }
        else {
            ROS_INFO("CHECK %f %f %f (%f)", x, y, z, sqrt(x*x + y*y + z*z));
        }
        //ROS_INFO("recv : %f %f %f", msg.x, msg.y, msg.z);
    }

    void callback(const sensor_msgs::Imu &imu_msg)
    {
        posefilter.SetData(imu_msg.header.stamp.toSec(), conv(imu_msg.linear_acceleration), conv(imu_msg.angular_velocity), resetIntegration);
        //tf::Vector3 p = tf::Vector3(0, 0, 0);
        tf::Vector3 p = posefilter.GetPosition();
        tf::Quaternion q = posefilter.GetPose(PoseFilter::COMP_FILTER);
        //tf::Quaternion q = posefilter.GetPose(PoseFilter::GYRO);
        resetIntegration = false;
        //geometry_msgs::Vector3 msg;
        //sst.publish(msg);
        //sst2.publish(conv(integ_omega));
        //info("cur", q);

        geometry_msgs::TransformStamped tfs;
        tfs.header.stamp = ros::Time::now();
        tfs.header.frame_id = "base";
        tfs.child_frame_id  = "realsense";
        //tfs.transform.translation = conv(p); // 重力方向のバイアスが十分取り除けてない。速度成分の誤差累積が累積してしまう。パラメーター調整して閾値以下には反応しないようにするか、一定速度仮定を用いるかした方がいいかも
        tfs.transform.translation = conv(tf::Vector3(0, 0, 0));
        tfs.transform.rotation = conv(q);
        tf_broadcaster.sendTransform(tfs);
    }

    void exec()
    {
        ros::spin();
    }
};



int main(int argc, char **argv) {
    ros::init(argc, argv, "posefilter");
    D455PoseFilter d455;
    d455.exec();
    return 0;
}
