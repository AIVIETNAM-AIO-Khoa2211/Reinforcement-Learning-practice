#include <iostream>
#include <random>
#include <string>
#include <iomanip>
#include <utility>   // Để dùng std::pair và std::make_pair
#include <algorithm> // Để dùng std::sort
#include <vector>

using namespace std;

//Khởi tạo engine sinh số ngẫu nhiên Mersenne Twister
random_device rd;
mt19937 generator(rd());
const string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 0123456789 .,!?'";
int get_random_int(int min_val, int max_val) {
    uniform_int_distribution<int> dist(min_val, max_val);
    return dist(generator);
}

//Step 1: tạo 1 cộng đồng có 100 cá thể
vector<string> generate_n_string(int len, int n) {
    vector<string> result;
    result.reserve(n); 

    // 1. Giới hạn index ngẫu nhiên từ 0 đến charset.length() - 1
    uniform_int_distribution<int> distribution(0, charset.length() - 1);

    for (int i = 0; i < n; i++) {
        string temp = "";
        temp.reserve(len); 
        
        for (int j = 0; j < len; j++) {
            int ran_idx = distribution(generator);
            temp += charset[ran_idx];
        }
        
        result.push_back(temp);
    }
    
    return result;
}

//Step 2: Tính điểm cho cá thể
double cal_fitness(string target, string tester) {
    int len = target.size();
    int score = 0;

    for (int i = 0; i < len; i++) {
        if (target[i]==tester[i]) score++;
    }
    return double(score)/len;
}

//Step 3: Chọn ra 2 cá thể mạnh nhất
pair<string, string> selection(const vector<string>& population, const vector<double>& scores) {
    int pop_size = population.size();
    
    // 1. Gộp cá thể và điểm số tương ứng thành một pair (score, individual)
    vector<pair<double, string>> candidate_list(pop_size);
    for (int i = 0; i < pop_size; ++i) {
        candidate_list[i] = {scores[i], population[i]};
    }

    // 2. Sắp xếp giảm dần theo điểm số (score)
    // std::sort mặc định so sánh element đầu tiên của pair (chính là score)
    sort(candidate_list.begin(), candidate_list.end(), [](const auto& a, const auto& b) {
        return a.first > b.first; 
    });

    // 3. Trả về 2 cá thể ở vị trí top 1 và top 2 có điểm cao nhất
    return make_pair(candidate_list[0].second, candidate_list[1].second);
}

//Step 4: Lai 2 cá thể mạnh nhất
string crossover (string par1, string par2) {
    int len = par1.size();
    int ran_cut_point = get_random_int(1, len -1);
    //Lấy từ đầu của A ghép với phần sau của B
    string child = par1.substr(0,ran_cut_point) + par2.substr(ran_cut_point);
    return child;
    
}

//Step 5: Đột biến đứa con
void mutate (string& child, double mutation_rate = 0.05) {
    int len =  child.size();
    for (int i = 0; i < len; i++) {
        double temp = get_random_int(1,100)/100.;
        if (temp<mutation_rate) { 
            int ran = get_random_int(0, charset.length() - 1);
            child[i] = charset[ran];
        }
    }
}

//Step 6: Vòng lặp chính
void main_evolution_loop(vector<string>& population, vector<double>& scores, string target) {
    cout << "=== BAT DAU LAI TAO ===" << endl;

    int pop_size = population.size();
    int generation = 0;
    int len = target.size();
    double max_score = 0;

    // 1. Tạo danh sách cá thể
    vector<pair<double, string>> candidate_list(pop_size);
    for (int i = 0; i < pop_size; ++i) {
        candidate_list[i] = {scores[i], population[i]};
    }

    // 2. Vòng lặp tiến hóa
    while (max_score < 1.0) {
        generation++;

        // Sắp xếp giảm dần theo score để dễ lấy Top (Bố mẹ) và Bottom (Cá thể yếu)
        sort(candidate_list.begin(), candidate_list.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        // Cập nhật điểm max hiện tại
        max_score = candidate_list[0].first;

        // In ra cá thể tốt nhất của thế hệ này
        cout << "Gen " << setw(4) << generation 
             << " | Best: " << candidate_list[0].second 
             << " | Score: " << candidate_list[0].first << endl;

        if (max_score >= 1.0) {
            cout << "\n==> DA TIM THAY TARGET SAU " << generation << " THE HE!" << endl;
            break;
        }

        // Chọn 2 bố mẹ tốt nhất thế hệ hiện tại
        string father = candidate_list[0].second;
        string mother = candidate_list[1].second;

        // Thay thế 50% cá thể yếu nhất ở nửa sau danh sách bằng con của father & mother
        for (int i = pop_size / 2; i < pop_size; i++) {
            string child = crossover(father, mother);
            mutate(child, 0.05); // Mutation rate 5%
            
            double child_score = cal_fitness(target, child);
            candidate_list[i] = {child_score, child};
        }
    }
}

int main() {
    int n = 100;
    int min = 0;
    string target = "Congratulation! You've done something.";
    int len = target.size();

    vector<string> res = generate_n_string(len, n);
    vector<double> scores(n);
    
    cout << "=== DANH SACH CHUOI NGAU NHIEN ===" << endl;
    
    for (int i = 0; i < n; i++) {
        double score = cal_fitness(target, res[i]);
        scores[i] = scores[i];
        // setw(3): Dành cố định 3 khoảng trống cho STT
        // left: Căn trái cho chuỗi res[i]
        cout << setw(3) << i + 1  
             << "| " << left << setw(len) << res[i] 
             << " |" << "Score: "<< score << endl;
    }
    main_evolution_loop(res, scores,target);

    return 0;
}
