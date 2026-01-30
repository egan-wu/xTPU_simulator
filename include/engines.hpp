#pragma once
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include "commands.hpp"
#include "status_register.hpp"
#include "memory.hpp"

template <typename CmdType>
class Engine {
public:
    Engine(StatusRegister& status_reg)
        : status_reg(status_reg), running(true) {
        worker = std::thread(&Engine::loop, this);
    }

    virtual ~Engine() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            running = false;
        }
        cv.notify_one();
        if (worker.joinable()) worker.join();
    }

    void push_command(const CmdType& cmd) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            cmd_queue.push(cmd);
        }
        cv.notify_one();
    }

protected:
    StatusRegister& status_reg;

    virtual void process(const CmdType& cmd) = 0;

private:
    std::queue<CmdType> cmd_queue;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    bool running;

    void loop() {
        while (true) {
            CmdType cmd;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return !cmd_queue.empty() || !running; });

                if (!running && cmd_queue.empty()) return;

                cmd = cmd_queue.front();
                cmd_queue.pop();
            }

            process(cmd);
        }
    }
};

class SDMAEngine : public Engine<DMA_Command> {
public:
    SDMAEngine(StatusRegister& sr, Scratchpad& sp);
protected:
    void process(const DMA_Command& cmd) override;
private:
    Scratchpad& scratchpad;
};

class IDMAEngine : public Engine<DMA_Command> {
public:
    IDMAEngine(StatusRegister& sr, Scratchpad& sp, LocalMemory& lm0, LocalMemory& lm1);
protected:
    void process(const DMA_Command& cmd) override;
private:
    Scratchpad& scratchpad;
    LocalMemory& local_mem0;
    LocalMemory& local_mem1;
};

class ComputeEngine : public Engine<Compute_Command> {
public:
    ComputeEngine(StatusRegister& sr, LocalMemory& lm, uint32_t busy_bit);
protected:
    void process(const Compute_Command& cmd) override;
private:
    LocalMemory& local_mem;
    uint32_t my_busy_bit;
};
