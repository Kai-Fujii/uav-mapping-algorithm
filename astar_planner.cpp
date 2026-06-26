#include "cae_percep/astar_planner.hpp"

AStarPlanner::AStarPlanner(int cell_num_x, int cell_num_y, float traversable_threshold)
    : cell_num_x_(cell_num_x), cell_num_y_(cell_num_y), traversable_threshold_(traversable_threshold) {}

AStarPlanner::~AStarPlanner() {
    for (auto node : node_allocator_) {
        delete node;
    }
}

std::vector<AStarNode*> AStarPlanner::reconstructPath(AStarNode* node) {
    std::vector<AStarNode*> path;
    AStarNode* temp = node;
    while (temp != nullptr) {
        path.push_back(temp);
        temp = temp->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

PathResult AStarPlanner::findPath(
    int start_x, int start_y,
    int goal_x, int goal_y,
    const std::unordered_map<int, CellInfo>& cell_data)
{
    PathResult result;
    result.status = SearchResult::FAILED;

    // 既存のノードをクリア
    for (auto node : node_allocator_) {
        delete node;
    }
    node_allocator_.clear();

    std::priority_queue<AStarNode*, std::vector<AStarNode*>, CompareNode> open_list;
    std::unordered_set<int> closed_list;

    AStarNode* start_node = new AStarNode(start_x, start_y, 0, 0, nullptr);
    node_allocator_.push_back(start_node);
    open_list.push(start_node);

    // 8方向の移動
    int movements[8][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    while (!open_list.empty()) {
        AStarNode* current_node = open_list.top();  // 最小のf(=g+h)の値を持つノードを取得
        open_list.pop();

        int current_id = current_node->x * cell_num_y_ + current_node->y;
        if (closed_list.count(current_id)) {
            continue;
        }
        closed_list.insert(current_id);

        // ゴール判定
        if (current_node->x == goal_x && current_node->y == goal_y) {
            result.status = SearchResult::REACH_GOAL;
            result.path = reconstructPath(current_node);
            return result;
        }

        // 隣接ノードを展開
        for (auto& move : movements) {
            int next_x = current_node->x + move[0];
            int next_y = current_node->y + move[1];
            int next_id = next_x * cell_num_y_ + next_y;

            // 1. マップ範囲外または探索済みならスキップ
            if (next_x < 0 || next_x >= cell_num_x_ || next_y < 0 || next_y >= cell_num_y_ || closed_list.count(next_id)) {
                continue;
            }

            // 2. ★Receding Horizon: 未知領域に到達したらサブゴールとして返す
            if (!cell_data.count(next_id)) {
                // current_nodeは既知領域の最後のノード → これをサブゴールとする
                result.status = SearchResult::REACH_HORIZON;
                result.path = reconstructPath(current_node);
                return result;
            }

            // 3. 移動先セルが通行不可ならスキップ
            if (!cell_data.at(next_id).is_traversable) {
                continue;  // 通行不可
            }

            // 4. 隙間の高さが連続しているか
            float current_z = 0.0f;
            float next_z = 0.0f;
            if (cell_data.count(current_id) && cell_data.count(next_id)) {
                const auto& current_info = cell_data.at(current_id);
                const auto& next_info = cell_data.at(next_id);

                // 両方の隙間の重なりを計算
                float overlap_max = std::min(current_info.aisle_max, next_info.aisle_max);
                float overlap_min = std::max(current_info.aisle_min, next_info.aisle_min);
                float overlap = overlap_max - overlap_min;

                if (overlap < traversable_threshold_) {
                    continue;  // 隙間が連続していない
                }

                // 飛行高さを計算(上下コスト用)
                current_z = (current_info.aisle_max + current_info.aisle_min) / 2.0f;
                next_z = (next_info.aisle_max + next_info.aisle_min) / 2.0f;
            }
            //5. チェック通過後にノードを追加
            float move_cost = (move[0] == 0 || move[1] == 0) ? 1.0f : 1.414f;

            //上下コスト（水平0.1mあたりcost=1.0に対して、上下0.1mあたりcost=1.5になるよう設定）
            const float z_weight = 15.0f;  // 上下0.1m = cost 1.5
            float dz = std::abs(next_z - current_z);
            float z_cost = dz * z_weight;
            
            float g_cost = current_node->g + move_cost + z_cost;
            float h_cost = std::sqrt(std::pow(next_x - goal_x, 2) + std::pow(next_y - goal_y, 2));

            //
            AStarNode* next_node = new AStarNode(next_x, next_y, g_cost, h_cost, current_node);
            node_allocator_.push_back(next_node);
            open_list.push(next_node);
        }
    }

    return result;  // 経路が見つからなかった場合(FAILED)
}

//かかか