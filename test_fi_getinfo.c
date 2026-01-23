#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

int main() {
    struct fi_info* hints = NULL;
    struct fi_info* info_list = NULL;
    int ret;
    
    printf("Testing fi_getinfo() for CXI provider...\n\n");
    
    // Allocate hints
    hints = fi_allocinfo();
    if (!hints) {
        fprintf(stderr, "ERROR: Failed to allocate fi_info hints\n");
        return 1;
    }
    
    // Configure hints for CXI
    hints->caps = FI_MSG | FI_HMEM;
    hints->mode = FI_CONTEXT;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT | FI_MR_HMEM |
                                  FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->av_type = FI_AV_UNSPEC;
    // Set provider name (need to allocate memory)
    hints->fabric_attr->prov_name = strdup("cxi");
    
    printf("Calling fi_getinfo() with provider=cxi...\n");
    ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info_list);
    
    if (ret != 0) {
        printf("fi_getinfo v1.18 failed: %s (ret=%d)\n", fi_strerror(-ret), ret);
        printf("Trying v1.6...\n");
        ret = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0, hints, &info_list);
    }
    
    if (ret == 0 && info_list) {
        int count = 0;
        printf("\n✓ fi_getinfo succeeded!\n\n");
        printf("Found devices:\n");
        
        for (struct fi_info* cur = info_list; cur; cur = cur->next) {
            count++;
            printf("\nDevice %d:\n", count);
            if (cur->fabric_attr && cur->fabric_attr->prov_name) {
                printf("  Provider: %s\n", cur->fabric_attr->prov_name);
            }
            if (cur->domain_attr && cur->domain_attr->name) {
                printf("  Domain: %s\n", cur->domain_attr->name);
            }
            if (cur->fabric_attr && cur->fabric_attr->name) {
                printf("  Fabric: %s\n", cur->fabric_attr->name);
            }
            if (cur->domain_attr) {
                printf("  HMEM support: %s\n", 
                       (cur->domain_attr->mr_mode & FI_MR_HMEM) ? "yes" : "no");
            }
        }
        
        if (count == 0) {
            printf("  (No devices found)\n");
        } else {
            printf("\nTotal: %d device(s)\n", count);
        }
        
        fi_freeinfo(info_list);
    } else {
        printf("\n✗ fi_getinfo failed: %s (ret=%d)\n", fi_strerror(-ret), ret);
        printf("\nPossible reasons:\n");
        printf("  1. Not running on a compute node (CXI devices only on compute nodes)\n");
        printf("  2. CXI provider not available\n");
        printf("  3. libfabric not properly configured\n");
    }
    
    // Don't free prov_name - fi_freeinfo will handle it
    fi_freeinfo(hints);
    return (ret == 0 && info_list) ? 0 : 1;
}

