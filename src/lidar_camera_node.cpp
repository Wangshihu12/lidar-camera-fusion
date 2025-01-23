#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <armadillo>

#include <chrono>

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

// Publisher
ros::Publisher pcOnimg_pub;
ros::Publisher pc_pub;

float maxlen = 100.0; // maxima distancia del lidar
float minlen = 0.01;  // minima distancia del lidar
float max_FOV = 3.0;  // en radianes angulo maximo de vista de la camara
float min_FOV = 0.4;  // en radianes angulo minimo de vista de la camara

/// parametros para convertir nube de puntos en imagen
float angular_resolution_x = 0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;
double max_var = 50.0;

float interpol_value = 20.0;

bool f_pc = true;

// input topics
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic = "/velodyne_points";

// matrix calibration lidar and camera

Eigen::MatrixXf Tlc(3, 1); // translation matrix lidar-camera
Eigen::MatrixXf Rlc(3, 3); // rotation matrix lidar-camera
Eigen::MatrixXf Mc(3, 4);  // camera calibration matrix

// range image parametros
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

///////////////////////////////////////callback

// 这个回调函数处理同步的点云和图像数据
void callback(const boost::shared_ptr<const sensor_msgs::PointCloud2> &in_pc2, const ImageConstPtr &in_image)
{
  // 将ROS图像消息转换为OpenCV格式
  cv_bridge::CvImagePtr cv_ptr, color_pcl;
  try {
    cv_ptr = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
    color_pcl = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // 将ROS点云消息转换为PCL点云格式
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*in_pc2, pcl_pc2);
  PointCloud::Ptr msg_pointCloud(new PointCloud);
  pcl::fromPCLPointCloud2(pcl_pc2, *msg_pointCloud);

  // 检查点云是否为空
  if (msg_pointCloud == NULL)
    return;

  // 过滤点云数据
  PointCloud::Ptr cloud_in(new PointCloud);
  PointCloud::Ptr cloud_out(new PointCloud);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

  // 根据距离过滤点云
  for (int i = 0; i < (int)cloud_in->points.size(); i++) {
    double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);
    if (distance < minlen || distance > maxlen)
      continue;
    cloud_out->push_back(cloud_in->points[i]);
  }

  // 将点云转换为球面投影的深度图像
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                                   pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                                   sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

  // 创建深度图像矩阵
  int cols_img = rangeImage->width;
  int rows_img = rangeImage->height;
  arma::mat Z;  // 距离插值矩阵
  arma::mat Zz; // 高度插值矩阵
  Z.zeros(rows_img, cols_img);
  Zz.zeros(rows_img, cols_img);
  Eigen::MatrixXf ZZei(rows_img, cols_img);

  // 填充深度图像矩阵
  for (int i = 0; i < cols_img; ++i)
    for (int j = 0; j < rows_img; ++j) {
      float r = rangeImage->getPoint(i, j).range;
      float zz = rangeImage->getPoint(i, j).z;
      if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz)) {
        continue;
      }
      Z.at(j, i) = r;
      Zz.at(j, i) = zz;
    }

  // 对深度图像进行插值处理
  arma::vec X = arma::regspace(1, Z.n_cols);
  arma::vec Y = arma::regspace(1, Z.n_rows);
  arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
  arma::vec YI = arma::regspace(Y.min(), 1.0/interpol_value, Y.max());
  arma::mat ZI_near;
  arma::mat ZI;
  arma::mat ZzI;
  arma::interp2(X, Y, Z, XI, YI, ZI, "lineal");
  arma::interp2(X, Y, Zz, XI, YI, ZzI, "lineal");

  // 将插值后的深度图像重建为3D点云
  PointCloud::Ptr point_cloud(new PointCloud);
  PointCloud::Ptr cloud(new PointCloud);
  point_cloud->width = ZI.n_cols;
  point_cloud->height = ZI.n_rows;
  point_cloud->is_dense = false;
  point_cloud->points.resize(point_cloud->width * point_cloud->height);

  // 对插值结果进行过滤
  arma::mat Zout = ZI;
  // 过滤插值产生的背景点
  for (uint i = 0; i < ZI.n_rows; i += 1) {
    for (uint j = 0; j < ZI.n_cols; j += 1) {
      if ((ZI(i, j) == 0)) {
        if (i + interpol_value < ZI.n_rows)
          for (int k = 1; k <= interpol_value; k += 1)
            Zout(i + k, j) = 0;
        if (i > interpol_value)
          for (int k = 1; k <= interpol_value; k += 1)
            Zout(i - k, j) = 0;
      }
    }
  }

  // 根据方差进行过滤
  if (f_pc) {
    for (uint i = 0; i < ((ZI.n_rows - 1) / interpol_value); i += 1)
      for (uint j = 0; j < ZI.n_cols - 5; j += 1) {
        double promedio = 0;
        double varianza = 0;
        for (uint k = 0; k < interpol_value; k += 1)
          promedio = promedio + ZI((i * interpol_value) + k, j);
        promedio = promedio / interpol_value;

        for (uint l = 0; l < interpol_value; l++)
          varianza = varianza + pow((ZI((i * interpol_value) + l, j) - promedio), 2.0);

        if (varianza > max_var)
          for (uint m = 0; m < interpol_value; m++)
            Zout((i * interpol_value) + m, j) = 0;
      }
    ZI = Zout;
  }

  // 将深度图像转换回点云
  int num_pc = 0;
  for (uint i = 0; i < ZI.n_rows - interpol_value; i += 1) {
    for (uint j = 0; j < ZI.n_cols; j += 1) {
      float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));
      if (ang < min_FOV - M_PI/2.0 || ang > max_FOV - M_PI/2.0)
        continue;

      if (!(Zout(i, j) == 0)) {
        // 计算3D点坐标
        float pc_modulo = Zout(i, j);
        float pc_x = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * cos(ang);
        float pc_y = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * sin(ang);

        // 应用激光雷达校正矩阵
        float ang_x_lidar = 0.6 * M_PI / 180.0;
        Eigen::MatrixXf Lidar_matrix(3, 3);
        Eigen::MatrixXf result(3, 1);
        Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                        0, 1, 0,
                        -sin(ang_x_lidar), 0, cos(ang_x_lidar);

        result << pc_x, pc_y, ZzI(i, j);
        result = Lidar_matrix * result;

        point_cloud->points[num_pc].x = result(0);
        point_cloud->points[num_pc].y = result(1);
        point_cloud->points[num_pc].z = result(2);

        cloud->push_back(point_cloud->points[num_pc]);
        num_pc++;
      }
    }
  }

  // 点云投影到图像平面
  PointCloud::Ptr P_out = cloud;
  Eigen::MatrixXf RTlc(4, 4);
  RTlc << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
          Rlc(1), Rlc(4), Rlc(7), Tlc(1),
          Rlc(2), Rlc(5), Rlc(8), Tlc(2),
          0, 0, 0, 1;

  int size_inter_Lidar = (int)P_out->points.size();
  unsigned int cols = in_image->width;
  unsigned int rows = in_image->height;

  // 创建彩色点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_color(new pcl::PointCloud<pcl::PointXYZRGB>);

  // 将点云投影到图像上并为点云着色
  for (int i = 0; i < size_inter_Lidar; i++) {
    Eigen::MatrixXf pc_matrix(4, 1);
    pc_matrix << -P_out->points[i].y, -P_out->points[i].z, P_out->points[i].x, 1.0;

    Eigen::MatrixXf Lidar_cam = Mc * (RTlc * pc_matrix);
    uint px_data = (int)(Lidar_cam(0, 0) / Lidar_cam(2, 0));
    uint py_data = (int)(Lidar_cam(1, 0) / Lidar_cam(2, 0));

    if (px_data < 0.0 || px_data >= cols || py_data < 0.0 || py_data >= rows)
      continue;

    // 计算点云颜色
    int color_dis_x = (int)(255 * ((P_out->points[i].x) / maxlen));
    int color_dis_z = (int)(255 * ((P_out->points[i].x) / 10.0));
    if (color_dis_z > 255)
      color_dis_z = 255;

    // 获取图像颜色并创建彩色点云
    cv::Vec3b &color = color_pcl->image.at<cv::Vec3b>(py_data, px_data);
    pcl::PointXYZRGB point;
    point.x = P_out->points[i].x;
    point.y = P_out->points[i].y;
    point.z = P_out->points[i].z;
    point.r = (int)color[2];
    point.g = (int)color[1];
    point.b = (int)color[0];
    pc_color->points.push_back(point);

    // 在图像上绘制点云
    cv::circle(cv_ptr->image, cv::Point(px_data, py_data), 1,
               CV_RGB(255-color_dis_x, (int)(color_dis_z), color_dis_x), cv::FILLED);
  }

  // 设置彩色点云属性
  pc_color->is_dense = true;
  pc_color->width = (int)pc_color->points.size();
  pc_color->height = 1;
  pc_color->header.frame_id = "velodyne";

  // 发布结果
  pcOnimg_pub.publish(cv_ptr->toImageMsg());
  pc_pub.publish(pc_color);
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "pontCloudOntImage");
  ros::NodeHandle nh;

  /// Load Parameters

  nh.getParam("/maxlen", maxlen);
  nh.getParam("/minlen", minlen);
  nh.getParam("/max_ang_FOV", max_FOV);
  nh.getParam("/min_ang_FOV", min_FOV);
  nh.getParam("/pcTopic", pcTopic);
  nh.getParam("/imgTopic", imgTopic);
  nh.getParam("/max_var", max_var);
  nh.getParam("/filter_output_pc", f_pc);

  nh.getParam("/x_resolution", angular_resolution_x);
  nh.getParam("/y_interpolation", interpol_value);

  nh.getParam("/ang_Y_resolution", angular_resolution_y);

  XmlRpc::XmlRpcValue param;

  nh.getParam("/matrix_file/tlc", param);
  Tlc << (double)param[0], (double)param[1], (double)param[2];

  nh.getParam("/matrix_file/rlc", param);

  Rlc << (double)param[0], (double)param[1], (double)param[2], (double)param[3], (double)param[4], (double)param[5], (double)param[6], (double)param[7], (double)param[8];

  nh.getParam("/matrix_file/camera_matrix", param);

  Mc << (double)param[0], (double)param[1], (double)param[2], (double)param[3], (double)param[4], (double)param[5], (double)param[6], (double)param[7], (double)param[8], (double)param[9], (double)param[10], (double)param[11];

  message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 1);
  message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);

  typedef sync_policies::ApproximateTime<PointCloud2, Image> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
  rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);

  pc_pub = nh.advertise<PointCloud>("/points2", 1);

  ros::spin();
}
