#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        bool success=false;
        size_t addr;
        for (const auto& pair : free_block) 
            if(pair.second>=size)
            {
                size_t remainder=pair.second-size;
                size_t new_st=pair.first+size;
                addr=pair.first;
                free_block.erase(pair.first);
                if(remainder>0)
                    free_block[new_st]=remainder;
                success=true;
                break;
            }
        used+=size;
        if(success)
            return addr;
        addr=peak;
        peak+=size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        bool before=false,after=false;
        if(addr+size==peak)
            // 最后一节的释放单独处理，不放回free_block中
            peak=addr;
        else
        {
            free_block[addr]=size;
            map<size_t,size_t>::iterator it=free_block.find(addr);
            map<size_t,size_t>::iterator it2=std::next(it);
            if(it2!=free_block.end())
                if(addr+size==it2->first)
                {
                    free_block[addr]=size+it2->second;
                    free_block.erase(it2);
                }
            if(it!=free_block.begin())
            {
                it2=it;
                --it;
                if(it->first+it->second==it2->first)
                {
                    free_block[it->first]=it->second+it2->second;
                    free_block.erase(it2->first);
                }

            }
        }
            
            
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
