#pragma once
// P0-3: 移除 std::atomic，統一由 std::mutex 保護 busy_mask。
// 原設計同時使用 atomic + mutex，語意不清且 get_status() 繞過 mutex 直接讀 atomic。
// condition_variable 本身需要 unique_lock，mutex 是必須的；atomic 因此是多餘的。
//
// P2-4: 新增 STATUS_ERROR 處理
//   set_error(info) : 設定 STATUS_ERROR bit，記錄錯誤訊息（by any Engine catch block）
//   has_error()     : 查詢是否有未清除的 error
//   get_error_info(): 取得最後一次錯誤的描述字串
//   clear_error()   : 清除 STATUS_ERROR bit 和訊息（呼叫端負責）
//
// STATUS_ERROR 是 "sticky"：engine busy bits 由 BusyClearGuard 自動清除，
// 但 ERROR bit 必須由上層明確呼叫 clear_error() 才能清除，避免錯誤被無聲忽略。
#include <cstdint>
#include <string>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <bitset>
#include "common_types.hpp"

class StatusRegister {
public:
    StatusRegister() : busy_mask(0) {}

    void set_busy(uint32_t mask) {
        std::lock_guard<std::mutex> lock(mtx);
        uint32_t old = busy_mask;
        busy_mask |= mask;
        log_transition(old, busy_mask);
    }

    void clear_busy(uint32_t mask) {
        std::lock_guard<std::mutex> lock(mtx);
        uint32_t old = busy_mask;
        busy_mask &= ~mask;
        log_transition(old, busy_mask);
        cv.notify_all();
    }

    void wait_on_mask(uint32_t mask) {
        std::unique_lock<std::mutex> lock(mtx);
        // predicate 在 lock 持有狀態下評估，busy_mask 讀取是 thread-safe 的
        cv.wait(lock, [this, mask] {
            return (busy_mask & mask) == 0;
        });
    }

    // P0-3: get_status() 現在也在 mutex 保護下讀取，與其他方法語意一致
    uint32_t get_status() {
        std::lock_guard<std::mutex> lock(mtx);
        return busy_mask;
    }

    // ── P2-4: Error Bit API ──────────────────────────────────────────────────

    // Engine catch block 呼叫：設定 STATUS_ERROR，記錄錯誤訊息
    // 注意：error_info 僅保留最後一次錯誤；多重錯誤時資訊會被覆寫
    void set_error(const std::string& info) {
        std::lock_guard<std::mutex> lock(mtx);
        uint32_t old = busy_mask;
        busy_mask |= STATUS_ERROR;
        last_error_info_ = info;
        log_transition(old, busy_mask);
        cv.notify_all();  // 通知等待者（如有人 wait_on_mask 其他 bits）
    }

    // 查詢是否有未清除的錯誤
    bool has_error() {
        std::lock_guard<std::mutex> lock(mtx);
        return (busy_mask & STATUS_ERROR) != 0;
    }

    // 取得最後一次錯誤的描述（呼叫前建議先 has_error()）
    std::string get_error_info() {
        std::lock_guard<std::mutex> lock(mtx);
        return last_error_info_;
    }

    // 上層明確清除 ERROR bit 和訊息（sticky，不會自動清除）
    void clear_error() {
        std::lock_guard<std::mutex> lock(mtx);
        uint32_t old = busy_mask;
        busy_mask &= ~STATUS_ERROR;
        last_error_info_.clear();
        log_transition(old, busy_mask);
        cv.notify_all();
    }

private:
    uint32_t    busy_mask = 0;       // 由 mtx 保護，不需要 atomic
    std::string last_error_info_;    // P2-4: 最後一次錯誤的描述（sticky）
    std::mutex  mtx;
    std::condition_variable cv;

    // P2-4: bitset 寬度從 5 改為 6，顯示新增的 ERROR bit（bit 5）
    void log_transition(uint32_t old_val, uint32_t new_val) {
        std::cout << "[Scoreboard] [0b" << std::bitset<6>(old_val)
                  << "] -> [0b" << std::bitset<6>(new_val) << "]" << std::endl;
    }
};
