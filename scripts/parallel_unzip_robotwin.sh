#!/bin/bash

# Parallel RobotWin Dataset Unzip Script
# Efficiently unzips all ZIP files in RobotWin dataset with priority for test files

set -e

# Configuration
DATASET_ROOT="/dataset_rc_mm/share/datasets/huggingface.co/TianxingChen/RoboTwin2.0/dataset"
MAX_PARALLEL_JOBS=8  # Adjust based on your system
LOG_DIR="/tmp/robotwin_unzip_logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}🚀 Starting RobotWin Dataset Parallel Unzip${NC}"
echo -e "${BLUE}Dataset Root: ${DATASET_ROOT}${NC}"
echo -e "${BLUE}Max Parallel Jobs: ${MAX_PARALLEL_JOBS}${NC}"
echo -e "${BLUE}Log Directory: ${LOG_DIR}${NC}"
echo ""

# Function to unzip a single file
unzip_file() {
    local zip_file="$1"
    local task_dir="$2"
    local priority="$3"
    local log_file="${LOG_DIR}/$(basename "$zip_file" .zip).log"
    
    echo -e "${YELLOW}[$priority] Processing: $(basename "$zip_file")${NC}"
    
    {
        echo "=== Unzipping $zip_file ==="
        echo "Task Directory: $task_dir"
        echo "Start Time: $(date)"
        echo ""
        
        # Create temporary extraction directory
        local temp_dir="${task_dir}/temp_extract_$$"
        mkdir -p "$temp_dir"
        
        # Extract to temporary directory first
        if unzip -q "$zip_file" -d "$temp_dir"; then
            echo "✅ Extraction successful"
            
            # Move extracted contents to final location
            # Handle different possible structures
            if [ -d "$temp_dir/demo_clean" ]; then
                # Structure: temp_dir/demo_clean/data/*.hdf5
                mv "$temp_dir"/* "$task_dir/"
            elif [ -d "$temp_dir/data" ]; then
                # Structure: temp_dir/data/*.hdf5
                mkdir -p "$task_dir/demo_clean"
                mv "$temp_dir/data" "$task_dir/demo_clean/"
            else
                # Structure: temp_dir/*.hdf5 (flat)
                mkdir -p "$task_dir/demo_clean/data"
                find "$temp_dir" -name "*.hdf5" -exec mv {} "$task_dir/demo_clean/data/" \;
            fi
            
            # Clean up temporary directory
            rm -rf "$temp_dir"
            
            # Count extracted files
            local hdf5_count=$(find "$task_dir" -name "*.hdf5" | wc -l)
            echo "📊 Extracted $hdf5_count HDF5 files"
            echo "✅ Successfully processed: $(basename "$zip_file")"
            
        else
            echo "❌ Extraction failed for $zip_file"
            rm -rf "$temp_dir"
            return 1
        fi
        
        echo "End Time: $(date)"
        echo "=========================="
        
    } > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ [$priority] Completed: $(basename "$zip_file")${NC}"
    else
        echo -e "${RED}❌ [$priority] Failed: $(basename "$zip_file")${NC}"
        return 1
    fi
}

# Export function for parallel execution
export -f unzip_file
export LOG_DIR RED GREEN YELLOW BLUE NC

# Check if dataset directory exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo -e "${RED}❌ Dataset directory not found: $DATASET_ROOT${NC}"
    exit 1
fi

# Find all ZIP files and categorize them
echo -e "${BLUE}🔍 Scanning for ZIP files...${NC}"

declare -a test_files=()
declare -a other_files=()

while IFS= read -r -d '' zip_file; do
    task_dir=$(dirname "$zip_file")
    zip_name=$(basename "$zip_file")
    
    # Check if it's a test file (contains "test", "clean", or smaller size indicator)
    if [[ "$zip_name" =~ (test|clean|_50\.) ]]; then
        test_files+=("$zip_file|$task_dir")
    else
        other_files+=("$zip_file|$task_dir")
    fi
done < <(find "$DATASET_ROOT" -name "*.zip" -print0)

total_files=$((${#test_files[@]} + ${#other_files[@]}))
echo -e "${BLUE}📊 Found $total_files ZIP files${NC}"
echo -e "${BLUE}   - Priority files (test/clean): ${#test_files[@]}${NC}"
echo -e "${BLUE}   - Other files: ${#other_files[@]}${NC}"
echo ""

if [ $total_files -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No ZIP files found to extract${NC}"
    exit 0
fi

# Function to process files in parallel
process_files_parallel() {
    local files_array=("$@")
    local priority="$1"
    shift
    files_array=("$@")
    
    if [ ${#files_array[@]} -eq 0 ]; then
        return 0
    fi
    
    echo -e "${BLUE}🔄 Processing ${#files_array[@]} $priority files...${NC}"
    
    # Process files in parallel batches
    local batch_size=$MAX_PARALLEL_JOBS
    local total=${#files_array[@]}
    
    for ((i=0; i<total; i+=batch_size)); do
        local batch_end=$((i + batch_size - 1))
        if [ $batch_end -ge $total ]; then
            batch_end=$((total - 1))
        fi
        
        echo -e "${YELLOW}📦 Processing batch $((i/batch_size + 1)): files $((i+1))-$((batch_end+1))${NC}"
        
        # Start parallel jobs for this batch
        local pids=()
        for ((j=i; j<=batch_end; j++)); do
            local file_info="${files_array[j]}"
            local zip_file="${file_info%|*}"
            local task_dir="${file_info#*|}"
            
            # Start background job
            unzip_file "$zip_file" "$task_dir" "$priority" &
            pids+=($!)
        done
        
        # Wait for all jobs in this batch to complete
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait $pid; then
                ((failed++))
            fi
        done
        
        if [ $failed -gt 0 ]; then
            echo -e "${RED}⚠️  $failed files failed in this batch${NC}"
        fi
        
        echo -e "${GREEN}✅ Batch $((i/batch_size + 1)) completed${NC}"
        echo ""
    done
}

# Start timing
start_time=$(date +%s)

# Process priority files first (test/clean files)
if [ ${#test_files[@]} -gt 0 ]; then
    echo -e "${BLUE}🎯 Phase 1: Processing priority files (test/clean)${NC}"
    process_files_parallel "PRIORITY" "${test_files[@]}"
    echo -e "${GREEN}✅ Priority files completed${NC}"
    echo ""
fi

# Process other files
if [ ${#other_files[@]} -gt 0 ]; then
    echo -e "${BLUE}📦 Phase 2: Processing remaining files${NC}"
    process_files_parallel "REGULAR" "${other_files[@]}"
    echo -e "${GREEN}✅ Regular files completed${NC}"
    echo ""
fi

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

# Final summary
echo -e "${GREEN}🎉 All extractions completed!${NC}"
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "${BLUE}   - Total files processed: $total_files${NC}"
echo -e "${BLUE}   - Total time: ${minutes}m ${seconds}s${NC}"
echo -e "${BLUE}   - Logs available in: $LOG_DIR${NC}"

# Count total extracted HDF5 files
total_hdf5=$(find "$DATASET_ROOT" -name "*.hdf5" | wc -l)
echo -e "${BLUE}   - Total HDF5 files extracted: $total_hdf5${NC}"

# Show any failed extractions
failed_logs=$(find "$LOG_DIR" -name "*.log" -exec grep -l "❌ Extraction failed" {} \; 2>/dev/null || true)
if [ -n "$failed_logs" ]; then
    echo -e "${RED}⚠️  Some extractions failed. Check logs:${NC}"
    echo "$failed_logs"
else
    echo -e "${GREEN}✅ All extractions successful!${NC}"
fi

echo ""
echo -e "${GREEN}🚀 RobotWin dataset is ready for training!${NC}"

# Optional: Clean up ZIP files (uncomment if you want to remove them after extraction)
# echo -e "${YELLOW}🗑️  Cleaning up ZIP files...${NC}"
# find "$DATASET_ROOT" -name "*.zip" -delete
# echo -e "${GREEN}✅ ZIP files cleaned up${NC}"
