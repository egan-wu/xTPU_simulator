#pragma once
#include <cstdint>
#include <ostream>
#include <iostream>
#include <string>
#include <string_view>
#include <cstdlib>   // std::getenv
#include <mutex>

// ---------------------------------------------------------------------------
// Logger — 結構化日誌系統 (P4-2)
//
// 設計原則：
//   1. 格式：[tick] [LEVEL] [component] message
//   2. 可配置 log level：DEBUG / INFO / WARN / ERROR（遞增嚴重度）
//   3. 環境變數控制：XTPU_LOG_LEVEL=DEBUG/INFO/WARN/ERROR（預設 WARN）
//   4. 可重導向到任意 std::ostream（預設 std::cerr）
//   5. thread-safe：透過 global mutex 保護輸出（效能非 critical path）
//   6. Header-only：不增加 .cpp 依賴
//
// 使用方式：
//   // 取得 singleton
//   auto& log = Logger::instance();
//
//   // 輸出日誌
//   XTPU_LOG_DEBUG(log, "SDMA", sim.get_clock().now(), "Starting DMA transfer");
//   XTPU_LOG_INFO (log, "Sim",  sim.get_clock().now(), "Dispatch packet #42");
//   XTPU_LOG_WARN (log, "PU0",  sim.get_clock().now(), "Unexpected latency spike");
//   XTPU_LOG_ERROR(log, "IDMA", sim.get_clock().now(), "OOB access at 0x10000000");
//
//   // 重導向到檔案
//   std::ofstream f("sim.log"); log.set_output(f);
//
//   // 調整 level
//   log.set_level(Logger::Level::DEBUG);
// ---------------------------------------------------------------------------

class Logger {
public:
    enum class Level : uint8_t {
        DEBUG = 0,
        INFO  = 1,
        WARN  = 2,
        ERROR = 3,
        OFF   = 4   // 完全關閉
    };

    // Singleton（每個進程只需一個 logger instance）
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    // 設定最低輸出 level（低於此 level 的訊息被靜默丟棄）
    void set_level(Level lvl) { level_ = lvl; }
    Level get_level() const   { return level_; }

    // 重導向輸出到指定 stream（e.g. std::ofstream）
    void set_output(std::ostream& os) {
        std::lock_guard<std::mutex> lock(mtx_);
        out_ = &os;
    }

    // 主要輸出函式（通常透過巨集呼叫）
    void log(Level lvl, std::string_view component,
             uint64_t tick, std::string_view message) {
        if (lvl < level_) return;

        std::lock_guard<std::mutex> lock(mtx_);
        *out_ << "[" << tick << "] "
              << "[" << level_str(lvl) << "] "
              << "[" << component << "] "
              << message << "\n";
    }

    // 便利：直接用 Level 名稱呼叫（避免 switch-case 在 macro 中展開）
    bool is_enabled(Level lvl) const { return lvl >= level_; }

private:
    Logger() : out_(&std::cerr), level_(Level::WARN) {
        // P4-2: 環境變數覆蓋預設 level
        const char* env = std::getenv("XTPU_LOG_LEVEL");
        if (env) {
            std::string s(env);
            if (s == "DEBUG") level_ = Level::DEBUG;
            else if (s == "INFO")  level_ = Level::INFO;
            else if (s == "WARN")  level_ = Level::WARN;
            else if (s == "ERROR") level_ = Level::ERROR;
            else if (s == "OFF")   level_ = Level::OFF;
        }
    }

    static const char* level_str(Level lvl) {
        switch (lvl) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO:  return " INFO";
            case Level::WARN:  return " WARN";
            case Level::ERROR: return "ERROR";
            default:           return "  ???";
        }
    }

    std::ostream* out_;
    Level         level_;
    std::mutex    mtx_;
};

// ---------------------------------------------------------------------------
// 便利巨集（避免在 log 呼叫被 disable 時仍對 message 求值）
// ---------------------------------------------------------------------------
#define XTPU_LOG(logger, lvl, component, tick, msg) \
    do { if ((logger).is_enabled(lvl)) (logger).log(lvl, component, tick, msg); } while(0)

#define XTPU_LOG_DEBUG(logger, comp, tick, msg) \
    XTPU_LOG(logger, Logger::Level::DEBUG, comp, tick, msg)
#define XTPU_LOG_INFO(logger, comp, tick, msg) \
    XTPU_LOG(logger, Logger::Level::INFO,  comp, tick, msg)
#define XTPU_LOG_WARN(logger, comp, tick, msg) \
    XTPU_LOG(logger, Logger::Level::WARN,  comp, tick, msg)
#define XTPU_LOG_ERROR(logger, comp, tick, msg) \
    XTPU_LOG(logger, Logger::Level::ERROR, comp, tick, msg)
