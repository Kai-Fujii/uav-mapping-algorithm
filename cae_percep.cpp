#include <functional>
#include <memory>
#include <time.h>
#include <stdlib.h>
#include <csignal>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set> 
#include <mutex> 

// #include "xlsxwriter.h"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/path.hpp> 
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <tf2/convert.h>

#include "cae_percep/astar_planner.hpp" 

#define RED_TEXT "\033[31m"
#define BOLD_TEXT "\033[1m"
#define RESET_FORMAT "\033[0m"

#include "sensor_msgs/msg/image.hpp"
using std::placeholders::_1;

// --- 色変換関数 ---
void hsv2rgb(float h, float s, float v, int &_r, int &_g, int &_b) {
  float r = static_cast<float>(v);
  float g = static_cast<float>(v);
  float b = static_cast<float>(v);
  if (s > 0.0f) {
      h *= 6.0f;
      const int i = (int) h;
      const float f = h - (float) i;
      switch (i) {
          default:
          case 0:
              g *= 1 - s * (1 - f);
              b *= 1 - s;
              break;
          case 1:
              r *= 1 - s * f;
              b *= 1 - s;
              break;
          case 2:
              r *= 1 - s;
              b *= 1 - s * (1 - f);
              break;
          case 3:
              r *= 1 - s;
              g *= 1 - s * f;
              break;
          case 4:
              r *= 1 - s * (1 - f);
              g *= 1 - s;
              break;
          case 5:
              g *= 1 - s;
              b *= 1 - s * f;
              break;
      }
  }
  _r = static_cast<int>(r * 255);
  _g = static_cast<int>(g * 255);
  _b = static_cast<int>(b * 255);
}

class pointcloud_subscriber : public rclcpp::Node
{
public:
    std::string gng_main_frame_id;
    std::string gng_crop_frame_id;
    double gng_crop_min_x;
    double gng_crop_max_x;
    double gng_crop_min_y;
    double gng_crop_max_y;
    double gng_crop_min_z;
    double gng_crop_max_z;

    static constexpr float MAP_SIZE_X = 500.0f;      // lx(x方向全体長さ)
    static constexpr float MAP_SIZE_Y = 500.0f;      // ly(y方向全体長さ)
    static constexpr float CELL_SIZE = 0.2f;         // d(セル一辺の長さ)
    static constexpr int CELL_NUM_X = static_cast<int>(MAP_SIZE_X / CELL_SIZE);  //x方向セル数
    static constexpr int CELL_NUM_Y = static_cast<int>(MAP_SIZE_Y / CELL_SIZE);  //y方向セル数
    static constexpr float DRONE_RADIUS = 0.4f;      // dr(ドローン半径)
    static constexpr float TRAVERSABLE_THRESHOLD = 0.4f;  // ドローン高さ
    static constexpr float OPTIMISTIC_RADIUS = 1.0f;  // ドローン周辺1.0m
    static constexpr float OPTIMISTIC_RADIUS_SQ = OPTIMISTIC_RADIUS * OPTIMISTIC_RADIUS;


    pointcloud_subscriber()
    : Node("pointcloud_subscriber")
    {
        // ★初期化時に明示的にクリア
        z_cell_global_.clear();
        cell_data_global_.clear();
        accumulated_updated_cells_.clear();
        
        declare_parameter("pointCloudTopic", "/cloud_registered");
        get_parameter("pointCloudTopic", pointCloudTopic);                                  
        

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("voxel_markers", 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planned_path", 10);  // 経路トピックパブリッシャー

        gng_main_frame_id = this->declare_parameter<std::string>("gng_main_frame_id","map");
        gng_crop_frame_id = this->declare_parameter<std::string>("gng_crop_frame_id","body");
        
        gng_crop_min_x = this->declare_parameter<double>("gng_crop_min_x", 0.2);
        gng_crop_max_x = this->declare_parameter<double>("gng_crop_max_x", 999.9);
        gng_crop_min_y = this->declare_parameter<double>("gng_crop_min_y", -999.9);
        gng_crop_max_y = this->declare_parameter<double>("gng_crop_max_y", 999.9);
        gng_crop_min_z = this->declare_parameter<double>("gng_crop_min_z", -999.9);
        gng_crop_max_z = this->declare_parameter<double>("gng_crop_max_z", 1.0);

        // ライダーオフセットのパラメータ読み込み
        lidar_offset_x_ = this->declare_parameter<double>("lidar_offset_x", 0.21);
        lidar_offset_y_ = this->declare_parameter<double>("lidar_offset_y", 0.0);
        lidar_offset_z_ = this->declare_parameter<double>("lidar_offset_z", 0.1);

        rclcpp::QoS sensor_qos(10);
        sensor_qos.best_effort();

        // 1. トピック受信
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          pointCloudTopic, sensor_qos, std::bind(&pointcloud_subscriber::topic_callback, this, _1));

        colored_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("colored_cloud", 10);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ゴール受信
        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
          "astar_goal", 10,
          [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
              std::lock_guard<std::mutex> lock(mutex_);
              current_goal_ = *msg;
              goal_received_ = true;
              RCLCPP_INFO(this->get_logger(), "Goal received!");
          }
       );

        planner_ = std::make_unique<AStarPlanner>(CELL_NUM_X, CELL_NUM_Y, TRAVERSABLE_THRESHOLD);

        // 2. タイマー作成
        map_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500), 
            std::bind(&pointcloud_subscriber::map_timer_callback, this));

        plan_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000), 
            std::bind(&pointcloud_subscriber::plan_timer_callback, this));
    }

    std::string pointCloudTopic;

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;  // 経路トピックパブリッシャー
    
    rclcpp::TimerBase::SharedPtr map_timer_;
    rclcpp::TimerBase::SharedPtr plan_timer_;

    std::unique_ptr<AStarPlanner> planner_;

    std::mutex mutex_; 
    std::unordered_map<int, std::vector<float>> z_cell_global_;
    std::unordered_map<int, CellInfo> cell_data_global_;
    std::set<int> accumulated_updated_cells_; 

    double current_drone_x_ = 0.0;
    double current_drone_y_ = 0.0;
    double current_drone_z_ = 0.0;

    double lidar_offset_x_;
    double lidar_offset_y_;
    double lidar_offset_z_;

    //TFが正常かどうかのフラグ
    bool is_drone_pose_valid_ = false;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    geometry_msgs::msg::PoseStamped current_goal_;
    bool goal_received_ = false;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ==========================================
    // 役割1：情報の収集（高速・高頻度）
    // ==========================================
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        static int callback_count = 0;
        callback_count++;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Callback #%d: Received %d points", 
                            callback_count, msg->width * msg->height);

        geometry_msgs::msg::TransformStamped t;
        try {
          t = tf_buffer_->lookupTransform(gng_main_frame_id, gng_crop_frame_id, tf2::TimePointZero, tf2::durationFromSec(1.0));
        } catch (const tf2::TransformException &ex) { 
            RCLCPP_WARN(this->get_logger(), "TF1 failed: %s", ex.what());
            std::lock_guard<std::mutex> lock(mutex_);
            is_drone_pose_valid_ = false;  //TF取得失敗
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            current_drone_x_ = t.transform.translation.x;
            current_drone_y_ = t.transform.translation.y;
            current_drone_z_ = t.transform.translation.z;
            is_drone_pose_valid_ = true;  //TF取得成功
        }

        // const double process_range_x = 4.0; // 正確な範囲のみ処理
        // const double process_range_y = 4.0;

        geometry_msgs::msg::TransformStamped gng_main_tf_stamp;
        std::string fromFrameRel= gng_main_frame_id;
        std::string toFrameRel= msg->header.frame_id.c_str();
        try {
          gng_main_tf_stamp = tf_buffer_->lookupTransform(fromFrameRel, toFrameRel, tf2::TimePointZero);	
        } catch (const tf2::TransformException & ex) { 
            RCLCPP_WARN(this->get_logger(), "TF2 failed: %s -> %s (%s)", 
                        fromFrameRel.c_str(), toFrameRel.c_str(), ex.what());
            return; 
        }
        tf2::Quaternion gng_main_quat(gng_main_tf_stamp.transform.rotation.x, gng_main_tf_stamp.transform.rotation.y, gng_main_tf_stamp.transform.rotation.z, gng_main_tf_stamp.transform.rotation.w);
        tf2::Matrix3x3 gng_main_mat33(gng_main_quat);
        
        geometry_msgs::msg::TransformStamped gng_crop_tf_stamp;
        fromFrameRel= gng_crop_frame_id;
        toFrameRel= msg->header.frame_id.c_str();
        try {
          gng_crop_tf_stamp = tf_buffer_->lookupTransform(fromFrameRel, toFrameRel, tf2::TimePointZero);	
        } catch (const tf2::TransformException & ex) { 
            RCLCPP_WARN(this->get_logger(), "TF3 failed: %s -> %s (%s)", 
                        fromFrameRel.c_str(), toFrameRel.c_str(), ex.what());
            return; 
        }
        tf2::Quaternion gng_crop_quat(gng_crop_tf_stamp.transform.rotation.x, gng_crop_tf_stamp.transform.rotation.y, gng_crop_tf_stamp.transform.rotation.z, gng_crop_tf_stamp.transform.rotation.w);
        tf2::Matrix3x3 gng_crop_mat33(gng_crop_quat);

        RCLCPP_INFO_ONCE(this->get_logger(), "TF transforms OK, starting point processing");
        
        sensor_msgs::PointCloud2ConstIterator<float> iter_x1(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y1(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z1(*msg, "z");

        // 排他制御開始
        std::lock_guard<std::mutex> lock(mutex_);

        int processed_points = 0;

        for(; iter_x1 != iter_x1.end(); ++iter_x1, ++iter_y1, ++iter_z1)  
        {
            if(*iter_x1 == 0 && *iter_y1 == 0 && *iter_z1 == 0) continue;
            float x = *iter_x1; float y = *iter_y1; float z = *iter_z1;

            // 1. gng_crop_frame_id座標系での点の位置を計算
            double crop_frame_pcd_x = x*gng_crop_mat33[0][0]+y*gng_crop_mat33[0][1]+z*gng_crop_mat33[0][2]+gng_crop_tf_stamp.transform.translation.x;
            double crop_frame_pcd_y = x*gng_crop_mat33[1][0]+y*gng_crop_mat33[1][1]+z*gng_crop_mat33[1][2]+gng_crop_tf_stamp.transform.translation.y;
            double crop_frame_pcd_z = x*gng_crop_mat33[2][0]+y*gng_crop_mat33[2][1]+z*gng_crop_mat33[2][2]+gng_crop_tf_stamp.transform.translation.z;
            
            // 2. ドローン中心からの相対位置（オフセット考慮）
            // gng_crop_frame_id = ライダー位置なので、ドローン中心への変換が必要
            double relative_x = crop_frame_pcd_x - lidar_offset_x_;
            double relative_y = crop_frame_pcd_y - lidar_offset_y_;
            double relative_z = crop_frame_pcd_z - lidar_offset_z_;
            
            // 3. 距離の2乗を計算（sqrt不要で高速）
            double dist_sq = relative_x * relative_x + relative_y * relative_y + relative_z * relative_z;
            
            // 4. ドローン半径内の点を除外
            if (dist_sq < DRONE_RADIUS * DRONE_RADIUS) continue;
            
            // 5. 既存のクロップ処理
            if(crop_frame_pcd_x < gng_crop_min_x || crop_frame_pcd_x > gng_crop_max_x) continue;
            if(crop_frame_pcd_y < gng_crop_min_y || crop_frame_pcd_y > gng_crop_max_y) continue;
            if(crop_frame_pcd_z < gng_crop_min_z || crop_frame_pcd_z > gng_crop_max_z) continue;

            // 6. メイン座標系へ変換
            float x1 = x*gng_main_mat33[0][0]+y*gng_main_mat33[0][1]+z*gng_main_mat33[0][2]+gng_main_tf_stamp.transform.translation.x;
            float y1 = x*gng_main_mat33[1][0]+y*gng_main_mat33[1][1]+z*gng_main_mat33[1][2]+gng_main_tf_stamp.transform.translation.y;
            float z1 = x*gng_main_mat33[2][0]+y*gng_main_mat33[2][1]+z*gng_main_mat33[2][2]+gng_main_tf_stamp.transform.translation.z;

            
           // if (std::abs(x1 - current_drone_x_) > process_range_x / 2.0 || 
             //   std::abs(y1 - current_drone_y_) > process_range_y / 2.0) continue;

            int xc = static_cast<int>(std::round((x1 + MAP_SIZE_X / 2) / CELL_SIZE));
            int yc = static_cast<int>(std::round((y1 + MAP_SIZE_Y / 2) / CELL_SIZE));

            if(xc >= 0 && xc < CELL_NUM_X && yc >= 0 && yc < CELL_NUM_Y){
                int cell_id1 = xc * CELL_NUM_Y + yc;
                
                // 1cm刻みの解像度でフィルタリング（メモリ爆発防止）
                auto& points_in_cell = z_cell_global_[cell_id1];
                bool redundant = false;
                
                // すでに登録済みの点と比較
                if (!points_in_cell.empty()) {
                // 二分探索で z1 に最も近い値を探す
                   auto it = std::lower_bound(points_in_cell.begin(), points_in_cell.end(), z1);
                   // 前後の要素を確認（最大2回の比較で済む）
                   if (it != points_in_cell.end() && std::abs(*it - z1) < 0.01f) {
                         redundant = true;
                   }
                   if (!redundant && it != points_in_cell.begin()) {
                       auto prev = std::prev(it);
                       if (std::abs(*prev - z1) < 0.01f) {
                             redundant = true;
                       }
                   }
                }
                // 新しい高さの情報（1cm以上離れている点）なら追加
                if (!redundant) {
                     auto insert_pos = std::lower_bound(points_in_cell.begin(), points_in_cell.end(), z1);
                     points_in_cell.insert(insert_pos, z1);
                     accumulated_updated_cells_.insert(cell_id1);
                }
            }
        }
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Processed %d points, updated %zu cells", 
                            processed_points, accumulated_updated_cells_.size());
    }

    // ==========================================
    // 役割2：地図更新（0.5秒ごとに実行）
    // ==========================================
    void map_timer_callback()
    {
        // ★排他制御
        std::lock_guard<std::mutex> lock(mutex_);

        // if (is_drone_pose_valid_) {
        //     int drone_xc = static_cast<int>(std::round((current_drone_x_ + MAP_SIZE_X / 2) / CELL_SIZE));
        //     int drone_yc = static_cast<int>(std::round((current_drone_y_ + MAP_SIZE_Y / 2) / CELL_SIZE));
            
        //     const int radius_cells = static_cast<int>(std::ceil(OPTIMISTIC_RADIUS / CELL_SIZE));
        //     const float radius_cells_sq = (OPTIMISTIC_RADIUS / CELL_SIZE) * (OPTIMISTIC_RADIUS / CELL_SIZE);
            
        //     for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
        //         for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
        //             float dist_sq = static_cast<float>(dx * dx + dy * dy);
        //             if (dist_sq > radius_cells_sq) continue;
                    
        //             int nx = drone_xc + dx;
        //             int ny = drone_yc + dy;
        //             if (nx < 0 || nx >= CELL_NUM_X || ny < 0 || ny >= CELL_NUM_Y) continue;
                    
        //             int cell_id = nx * CELL_NUM_Y + ny;
                    
        //             // ★未観測セルのみ楽観的評価
        //             if (!cell_data_global_.count(cell_id)) {
        //                 CellInfo optimistic_info;
        //                 optimistic_info.is_traversable = true;
        //                 optimistic_info.aisle_max = current_drone_z_ + 1.0f;
        //                 optimistic_info.aisle_min = current_drone_z_ - 1.0f;
        //                 optimistic_info.max_diff = 2.0f;  // 十分な高さ
        //                 optimistic_info.max_point = current_drone_z_ + 1.0f;
        //                 optimistic_info.min_point = current_drone_z_ - 1.0f;
        //                 cell_data_global_[cell_id] = optimistic_info;
        //                 accumulated_updated_cells_.insert(cell_id);  // ★Loop2で処理されるように追加
        //             }
        //         }
        //     }
        // }

        // if (accumulated_updated_cells_.empty()) return;


        // Loop 1
        for (int cell_id : accumulated_updated_cells_) {
            std::vector<float>& zic = z_cell_global_[cell_id];

            // ★実際の点群がないセルはスキップ
            if (zic.empty()) continue;

            std::sort(zic.begin(), zic.end());
            float max_diff = 0.0f;
            float max_point = 0.0f;
            float min_point = 0.0f;

            if (!zic.empty() && is_drone_pose_valid_) {
                float highest_point = zic.back();
                float lowest_point = zic.front();
                // ドローンの現在高さを取得（z座標も必要になるので追加取得）
                float drone_z = current_drone_z_;

                    // 取得した点群が全てドローンより低い場合（床のみ観測）
                    if (highest_point < drone_z) {
                        min_point = highest_point;  // 最高点を床とする
                        max_point = drone_z + 0.5f; // ドローン高さ + 0.5m を仮の天井とする
                        max_diff = max_point - min_point;
                     }
                    // 取得した点群が全てドローンより高い場合（天井のみ観測）
                    else if (lowest_point > drone_z) {
                         max_point = lowest_point;    // 最低点を天井とする
                         min_point = drone_z - 0.65f; // ドローン高さ - 0.65m を仮の床とする
                         max_diff = max_point - min_point;
                    }
                    // 通常ケース：天井と床の両方が観測されている
                    else {
                        for (size_t i = 1; i < zic.size(); ++i) {
                            float diff = std::abs(zic[i] - zic[i - 1]);
                            if (diff > max_diff) {
                                max_diff = diff;
                                max_point = zic[i];
                                min_point = zic[i - 1];
                            }
                        }
                    }
                }

                CellInfo info;
                info.max_diff = max_diff;
                info.max_point = max_point;
                info.min_point = min_point;
                info.aisle_max = std::numeric_limits<float>::max();
                info.aisle_min = std::numeric_limits<float>::lowest();
                info.is_traversable = false;
                cell_data_global_[cell_id] = info;
            }

        // Loop 2
        const float drone_radius_cells = DRONE_RADIUS / CELL_SIZE;
        const float search_radius_cells = drone_radius_cells + 0.707f;
        const float search_radius_cells_sq = search_radius_cells * search_radius_cells;
        const int aisle_num = static_cast<int>(std::ceil(DRONE_RADIUS / CELL_SIZE));

        for (int cell_id : accumulated_updated_cells_) {
            int center_xc = cell_id / CELL_NUM_Y;
            int center_yc = cell_id % CELL_NUM_Y;
            float aisle_max = std::numeric_limits<float>::max();
            float aisle_min = std::numeric_limits<float>::lowest();
            bool has_data = false;

            for (int dx = -aisle_num; dx <= aisle_num; ++dx) {
                for (int dy = -aisle_num; dy <= aisle_num; ++dy) {
                    // ★平方根なしで距離チェック
                    float dist_sq = static_cast<float>(dx * dx + dy * dy);
                    if (dist_sq > search_radius_cells_sq) continue;

                    int nx = center_xc + dx; int ny = center_yc + dy;
                    if (nx < 0 || nx >= CELL_NUM_X || ny < 0 || ny >= CELL_NUM_Y) continue;
                    float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy)) * CELL_SIZE;
                    if (dist > DRONE_RADIUS + CELL_SIZE * 0.707f) continue;

                    int aisle_cell_id = nx * CELL_NUM_Y + ny;
                    if (cell_data_global_.count(aisle_cell_id)) {
                        const auto& info = cell_data_global_.at(aisle_cell_id);
                        aisle_max = std::min(aisle_max, info.max_point);
                        aisle_min = std::max(aisle_min, info.min_point);
                        has_data = true;
                    }
                }
            }
            if (has_data && cell_data_global_.count(cell_id)) {
                cell_data_global_[cell_id].aisle_max = aisle_max;
                cell_data_global_[cell_id].aisle_min = aisle_min;
                cell_data_global_[cell_id].is_traversable = (aisle_max - aisle_min >= TRAVERSABLE_THRESHOLD);
            }
        }
        
        accumulated_updated_cells_.clear();

        if (!cell_data_global_.empty()) {
            sensor_msgs::msg::PointCloud2 cell_cloud_msg;
            cell_cloud_msg.header.frame_id = gng_main_frame_id;
            cell_cloud_msg.header.stamp = this->get_clock()->now();
            cell_cloud_msg.height = 1;
            cell_cloud_msg.width = static_cast<uint32_t>(cell_data_global_.size());
            cell_cloud_msg.is_dense = false;
            sensor_msgs::PointCloud2Modifier cell_mod(cell_cloud_msg);
            cell_mod.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                             "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                             "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                             "rgb", 1, sensor_msgs::msg::PointField::UINT32);
            cell_mod.resize(cell_cloud_msg.width);
            sensor_msgs::PointCloud2Iterator<float> it_x(cell_cloud_msg, "x");
            sensor_msgs::PointCloud2Iterator<float> it_y(cell_cloud_msg, "y");
            sensor_msgs::PointCloud2Iterator<float> it_z(cell_cloud_msg, "z");
            sensor_msgs::PointCloud2Iterator<uint32_t> it_rgb(cell_cloud_msg, "rgb");
            
            for (const auto &pair : cell_data_global_) {
                int cell_id = pair.first;
                const auto &info = pair.second;
                int cur_xc = cell_id / CELL_NUM_Y;
                int cur_yc = cell_id % CELL_NUM_Y;
                *it_x = static_cast<float>(cur_xc) * CELL_SIZE - MAP_SIZE_X / 2.0f;
                *it_y = static_cast<float>(cur_yc) * CELL_SIZE - MAP_SIZE_Y / 2.0f;
                *it_z = 0.0f;
                int r, g, b;
                
                // ★is_traversableの結果で色分け
                if (info.is_traversable) {
                    // 通行可能 → 青
                    hsv2rgb(240.0f / 360.0f, 1.0f, 1.0f, r, g, b); 
                } else {
                    // 通行不可 → 赤
                    hsv2rgb(0.0f, 1.0f, 1.0f, r, g, b); 
                }
                
                *it_rgb = (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(b);
                ++it_x; ++it_y; ++it_z; ++it_rgb;
            }
            colored_cloud_pub_->publish(cell_cloud_msg);
        }
    }



    // ==========================================
    // 役割3：経路計画（1.0秒ごとに実行）
    // ==========================================
    void plan_timer_callback()
    {
        // ★排他制御
        std::lock_guard<std::mutex> lock(mutex_);

        if (!goal_received_) return;
        
        int start_xc = static_cast<int>(std::round((current_drone_x_ + MAP_SIZE_X / 2) / CELL_SIZE));
        int start_yc = static_cast<int>(std::round((current_drone_y_ + MAP_SIZE_Y / 2) / CELL_SIZE));
        int start_id = start_xc * CELL_NUM_Y + start_yc;
        
        if (!cell_data_global_.count(start_id)) {
            CellInfo s_info; s_info.is_traversable = true; s_info.aisle_max = 2.0; s_info.aisle_min = 0.0;
            cell_data_global_[start_id] = s_info;
        } else { cell_data_global_[start_id].is_traversable = true; }

        int goal_xc = static_cast<int>(std::round((current_goal_.pose.position.x + MAP_SIZE_X / 2) / CELL_SIZE));
        int goal_yc = static_cast<int>(std::round((current_goal_.pose.position.y + MAP_SIZE_Y / 2) / CELL_SIZE));
        // int goal_id = goal_xc * cell_num_y + goal_yc;

        // 真のゴールまでの距離を計算（到達判定用）
        float dist_to_true_goal = std::sqrt(
          std::pow(current_drone_x_ - current_goal_.pose.position.x, 2) +
          std::pow(current_drone_y_ - current_goal_.pose.position.y, 2));

        // 真のゴールに到達した場合（0.1m以内）
        if (dist_to_true_goal < 0.1f) {
          RCLCPP_INFO(this->get_logger(), "Reached GOAL");
          goal_received_ = false;
          return;
        }

        if (start_xc >= 0 && start_xc < CELL_NUM_X && goal_xc >= 0 && goal_xc < CELL_NUM_X) {
            PathResult result = planner_->findPath(start_xc, start_yc, goal_xc, goal_yc, cell_data_global_);
            
            if (result.status != SearchResult::FAILED && !result.path.empty()) {
                // RCLCPP_INFO(this->get_logger(), "Path found: %zu nodes", path.size());
                visualization_msgs::msg::MarkerArray path_markers;
                visualization_msgs::msg::Marker line_marker;
                line_marker.header.frame_id = gng_main_frame_id;
                line_marker.header.stamp = this->get_clock()->now();
                line_marker.ns = "path"; line_marker.id = 0;
                line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
                line_marker.action = visualization_msgs::msg::Marker::ADD;
                line_marker.scale.x = 0.1;
                line_marker.color.g = 1.0f; line_marker.color.a = 1.0f;  // 緑
                line_marker.pose.orientation.w = 1.0;


                for (const auto& node : result.path) {
                    geometry_msgs::msg::Point p;
                    p.x = static_cast<float>(node->x) * CELL_SIZE - MAP_SIZE_X / 2.0f;
                    p.y = static_cast<float>(node->y) * CELL_SIZE - MAP_SIZE_Y / 2.0f;
                    int nid = node->x * CELL_NUM_Y + node->y;
                    if(cell_data_global_.count(nid)) p.z = (cell_data_global_[nid].aisle_max + cell_data_global_[nid].aisle_min)/2.0f;
                    else p.z = 1.0f;
                    line_marker.points.push_back(p);
                }
                path_markers.markers.push_back(line_marker);
                

                // === スタートマーカー（黄緑） ===
                visualization_msgs::msg::Marker start_marker;
                start_marker.header.frame_id = gng_main_frame_id;
                start_marker.header.stamp = this->get_clock()->now();
                start_marker.ns = "start"; start_marker.id = 0;
                start_marker.type = visualization_msgs::msg::Marker::SPHERE;
                start_marker.action = visualization_msgs::msg::Marker::ADD;
                start_marker.pose.position.x = static_cast<float>(start_xc) * CELL_SIZE - MAP_SIZE_X / 2.0f;
                start_marker.pose.position.y = static_cast<float>(start_yc) * CELL_SIZE - MAP_SIZE_Y / 2.0f;
                start_marker.pose.position.z = cell_data_global_.count(start_id) ? 
                    (cell_data_global_[start_id].aisle_max + cell_data_global_[start_id].aisle_min)/2.0f : 1.0f;
                start_marker.pose.orientation.w = 1.0;
                start_marker.scale.x = 0.4; start_marker.scale.y = 0.4; start_marker.scale.z = 0.4;
                start_marker.color.r = 0.5f; start_marker.color.g = 1.0f; start_marker.color.b = 0.0f; start_marker.color.a = 1.0f;  // 黄緑
                path_markers.markers.push_back(start_marker);

                // === ゴールマーカー（黄色） ===
                visualization_msgs::msg::Marker goal_marker;
                goal_marker.header.frame_id = gng_main_frame_id;
                goal_marker.header.stamp = this->get_clock()->now();
                goal_marker.ns = "goal"; goal_marker.id = 0;
                goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
                goal_marker.action = visualization_msgs::msg::Marker::ADD;
                goal_marker.pose.position.x = current_goal_.pose.position.x;
                goal_marker.pose.position.y = current_goal_.pose.position.y;
                goal_marker.pose.position.z = current_goal_.pose.position.z > 0.01 ? current_goal_.pose.position.z : 1.0f;
                goal_marker.pose.orientation.w = 1.0;
                goal_marker.scale.x = 0.4; goal_marker.scale.y = 0.4; goal_marker.scale.z = 0.4;
                goal_marker.color.r = 1.0f; goal_marker.color.g = 1.0f; goal_marker.color.b = 0.0f; goal_marker.color.a = 1.0f;  // 黄色
                path_markers.markers.push_back(goal_marker);

                if (result.status == SearchResult::REACH_HORIZON) {
                  visualization_msgs::msg::Marker subgoal_marker;
                  subgoal_marker.header.frame_id = gng_main_frame_id;
                  subgoal_marker.header.stamp = this->get_clock()->now();
                  subgoal_marker.ns = "subgoal"; subgoal_marker.id = 1;
                  subgoal_marker.type = visualization_msgs::msg::Marker::SPHERE;
                  subgoal_marker.action = visualization_msgs::msg::Marker::ADD;
                  auto* sg = result.path.back();
                  subgoal_marker.pose.position.x = static_cast<float>(sg->x) * CELL_SIZE - MAP_SIZE_X / 2.0f;
                  subgoal_marker.pose.position.y = static_cast<float>(sg->y) * CELL_SIZE - MAP_SIZE_Y / 2.0f;
                  int sg_id = sg->x * CELL_NUM_Y + sg->y;
                  subgoal_marker.pose.position.z = cell_data_global_.count(sg_id) ? 
                      (cell_data_global_[sg_id].aisle_max + cell_data_global_[sg_id].aisle_min)/2.0f : 1.0f;
                  subgoal_marker.pose.orientation.w = 1.0;
                  subgoal_marker.scale.x = 0.3; subgoal_marker.scale.y = 0.3; subgoal_marker.scale.z = 0.3;
                  subgoal_marker.color.r = 1.0f; subgoal_marker.color.g = 0.5f; subgoal_marker.color.a = 1.0f;
                  path_markers.markers.push_back(subgoal_marker);
                }
              


                // === 経路トピックの発行 ===
                marker_pub_->publish(path_markers);
                
                // ★追加: nav_msgs/Path をパブリッシュ（drone_controller用）
                nav_msgs::msg::Path nav_path;
                nav_path.header.frame_id = gng_main_frame_id;
                nav_path.header.stamp = this->get_clock()->now();
                
                for (const auto& node : result.path) {
                    geometry_msgs::msg::PoseStamped pose;
                    pose.header = nav_path.header;
                    pose.pose.position.x = static_cast<float>(node->x) * CELL_SIZE - MAP_SIZE_X / 2.0f;
                    pose.pose.position.y = static_cast<float>(node->y) * CELL_SIZE - MAP_SIZE_Y / 2.0f;
                    int nid = node->x * CELL_NUM_Y + node->y;
                    if(cell_data_global_.count(nid)) {
                        pose.pose.position.z = (cell_data_global_[nid].aisle_max + cell_data_global_[nid].aisle_min) / 2.0f;
                    } else {
                        pose.pose.position.z = 1.0f;
                    }
                    pose.pose.orientation.w = 1.0;
                    nav_path.poses.push_back(pose);
                }
                path_pub_->publish(nav_path);


            } else {
                // RCLCPP_WARN(this->get_logger(), "Path NOT found");
            }
        }
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pointcloud_subscriber>());
  rclcpp::shutdown();
  return 0;
}
