/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2018
**
**  \section sref Reference:
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas, 
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated 
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics, 23(1), 010504, 2018.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_utils.c

@brief   mcconfiguration and command line option processing unit
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>

#ifndef WIN32
  #include <sys/ioctl.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include "mcx_utils.h"
#include "mcx_const.h"
#include "mcx_shapes.h"
#include "mcx_core.h"

/**
 * Macro to load JSON keys
 */
#define FIND_JSON_KEY(id,idfull,parent,fallback,val) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? fallback : tmp->val) \
                     : tmp->val)

/**
 * Macro to load JSON object
 */
#define FIND_JSON_OBJ(id,idfull,parent) \
                    ((tmp=cJSON_GetObjectItem(parent,id))==0 ? \
                                ((tmp=cJSON_GetObjectItem(root,idfull))==0 ? NULL : tmp) \
                     : tmp)
#ifdef WIN32
         char pathsep='\\';
#else
         char pathsep='/';
#endif

/**
 * Macro to include unit name and line number in the error message
 */
#define MCX_ASSERT(a)  (!(a) && (mcx_error((a),"input error",__FILE__,__LINE__),1) );

#define MIN_HEADER_SIZE 348    /**< Analyze header size */
#define NII_HEADER_SIZE 352    /**< NIFTI header size */
#define GL_RGBA32F 0x8814

/**
 * Short command line options
 * If a short command line option is '-' that means it only has long/verbose option.
 * Array terminates with '\0'.
 */

const char shortopt[]={'h','i','f','n','t','T','s','a','g','b','-','z','u','H','P',
                 'd','r','S','p','e','U','R','l','L','-','I','-','G','M','A','E','v','D',
		 'k','q','Y','O','F','-','-','x','X','-','K','m','V','B','W','w','-','\0'};

/**
 * Long command line options
 * The length of this array must match the length of shortopt[], terminates with ""
 */

const char *fullopt[]={"--help","--interactive","--input","--photon",
                 "--thread","--blocksize","--session","--array",
                 "--gategroup","--reflect","--reflectin","--srcfrom0",
                 "--unitinmm","--maxdetphoton","--shapes","--savedet",
                 "--repeat","--save2pt","--printlen","--minenergy",
                 "--normalize","--skipradius","--log","--listgpu","--faststep",
                 "--printgpu","--root","--gpu","--dumpmask","--autopilot",
		 "--seed","--version","--debug","--voidtime","--saveseed",
		 "--replaydet","--outputtype","--outputformat","--maxjumpdebug",
                 "--maxvoidstep","--saveexit","--saveref","--gscatter","--mediabyte",
                 "--momentum","--specular","--bc","--workload","--savedetflag","--internalsrc",""};

/**
 * Output data types
 * x: fluence rate
 * f: fluence
 * e: energy deposit
 * j: jacobian for mua
 * p: scattering counts for computing Jacobians for mus
 */

const char outputtype[]={'x','f','e','j','p','m','\0'};

/**
 * Debug flags
 * R: debug random number generator
 * M: record photon movement and trajectory
 * P: show progress bar
 */

const char debugflag[]={'R','M','P','\0'};

/**
 * Recorded fields for detected photons
 * D: detector ID (starting from 1) [1]
 * S: partial scattering event count [#media]
 * P: partial path lengths [#media]
 * M: momentum transfer [#media]
 * X: exit position [3]
 * V: exit direction vector [3]
 * W: initial weight [1]
 */

const char saveflag[]={'D','S','P','M','X','V','W','\0'};

/**
 * Output file format
 * mc2: binary mc2 format to store fluence volume data
 * nii: output fluence in nii format
 * hdr: output volume in Analyze hdr/img format
 * ubj: output volume in unversal binary json format (not implemented)
 */

const char *outputformat[]={"mc2","nii","hdr","ubj","tx3",""};

/**
 * Boundary condition (BC) types
 * _: no condition (fallback to isreflect)
 * r: Fresnel boundary
 * a: total absorption BC
 * m: total reflection (mirror) BC
 * c: cylic BC
 */

const char boundarycond[]={'_','r','a','m','c','\0'};

/**
 * Source type specifier
 * User can specify the source type using a string
 */

const char *srctypeid[]={"pencil","isotropic","cone","gaussian","planar",
    "pattern","fourier","arcsine","disk","fourierx","fourierx2d","zgaussian",
    "line","slit","pencilarray","pattern3d",""};


/**
 * Media byte format
 * User can specify the source type using a string
 */

const unsigned int mediaformatid[]={1,2,4,100,101,102,103,104,0};
const char *mediaformat[]={"byte","short","integer","muamus_float","mua_float","muamus_half","asgn_byte","muamus_short",""};


/**
 * Flag to decide if parameter has been initialized over command line
 */
 
char flagset[256]={'\0'};

/**
 * @brief Initializing the simulation configuration with default values
 *
 * Constructor of the simulation configuration, initializing all field to default values
 */

void mcx_initcfg(Config *cfg){
     cfg->medianum=0;
     cfg->mediabyte=1;        /** expect 1-byte per medium index, use --mediabyte to set to 2 or 4 */
     cfg->detnum=0;
     cfg->dim.x=0;
     cfg->dim.y=0;
     cfg->dim.z=0;
     cfg->steps.x=1.f;
     cfg->steps.y=1.f;
     cfg->steps.z=1.f;
     cfg->nblocksize=64;      /** in theory, mcx can use min block size 32 because no communication between threads, but 64 seems to work the best */
     cfg->nphoton=0;
     cfg->nthread=(1<<14);    /** launch many threads to saturate the device to maximize throughput */
     cfg->isrowmajor=0;       /** default is Matlab array */
     cfg->maxgate=0;
     cfg->isreflect=1;
     cfg->isref3=1;
     cfg->isrefint=0;
     cfg->isnormalized=1;
     cfg->issavedet=1;        /** output detected photon data by default, use -d 0 to disable */
     cfg->respin=1;
     cfg->issave2pt=1;
     cfg->isgpuinfo=0;
     cfg->prop=NULL;
     cfg->detpos=NULL;
     cfg->vol=NULL;
     cfg->session[0]='\0';
     cfg->printnum=0;
     cfg->minenergy=0.f;
     cfg->flog=stdout;
     cfg->sradius=-2.f;
     cfg->rootpath[0]='\0';
     cfg->gpuid=0;
     cfg->issrcfrom0=0;
     cfg->unitinmm=1.f;
     cfg->isdumpmask=0;
     cfg->srctype=0;;         /** use pencil beam as default source type */
     cfg->maxdetphoton=1000000;
     cfg->maxjumpdebug=10000000;
     cfg->exportdebugdata=NULL;
     cfg->debugdatalen=0;
     cfg->autopilot=1;
     cfg->seed=0x623F9A9E;    /** default RNG seed, a big integer, with a hidden meaning :) */
     cfg->exportfield=NULL;
     cfg->exportdetected=NULL;
     cfg->energytot=0.f;
     cfg->energyabs=0.f;
     cfg->energyesc=0.f;
     /*cfg->his=(History){{'M','C','X','H'},1,0,0,0,0,0,0,1.f,{0,0,0,0,0,0,0}};*/   /** This format is only supported by C99 */
     memset(&cfg->his,0,sizeof(History));
     memcpy(cfg->his.magic,"MCXH",4);
     cfg->his.version=1;
     cfg->his.unitinmm=1.f;
     cfg->his.normalizer=1.f;
     cfg->his.respin=1;
     cfg->his.srcnum=1;
     cfg->savedetflag=0x5;
     cfg->his.savedetflag=cfg->savedetflag;
     cfg->shapedata=NULL;
     cfg->seeddata=NULL;
     cfg->maxvoidstep=1000;
     cfg->voidtime=1;
     cfg->srcpattern=NULL;
     cfg->srcnum=1;
     cfg->debuglevel=0;
     cfg->issaveseed=0;
     cfg->issaveexit=0;
     cfg->ismomentum=0;
     cfg->internalsrc=0;
     cfg->replay.seed=NULL;
     cfg->replay.weight=NULL;
     cfg->replay.tof=NULL;
     cfg->replay.detid=NULL;
     cfg->replaydet=0;
     cfg->seedfile[0]='\0';
     cfg->outputtype=otFlux;
     cfg->outputformat=ofMC2;
     cfg->detectedcount=0;
     cfg->runtime=0;
     cfg->faststep=0;
     cfg->srcdir.w=0.f;
     cfg->issaveref=0;
     cfg->isspecular=0;
     cfg->dx=cfg->dy=cfg->dz=NULL;
     cfg->gscatter=1e9;     /** by default, honor anisotropy for all scattering, use --gscatter to reduce it */
     memset(cfg->bc,0,8);
     memset(&(cfg->srcparam1),0,sizeof(float4));
     memset(&(cfg->srcparam2),0,sizeof(float4));
     memset(cfg->deviceid,0,MAX_DEVICE);
     memset(cfg->workload,0,MAX_DEVICE*sizeof(float));
     cfg->deviceid[0]='1';  /** use the first GPU device by default*/
#ifdef MCX_CONTAINER
     cfg->parentid=mpMATLAB;
#else
     cfg->parentid=mpStandalone;
#endif
}

/**
 * @brief Reset and clear the GPU information data structure
 *
 * Clearing the GPU information data structure
 */

void mcx_cleargpuinfo(GPUInfo **gpuinfo){
    if(*gpuinfo){
	free(*gpuinfo);
	*gpuinfo=NULL;
    }
}

/**
 * @brief Clearing the simulation configuration data structure
 *
 * Destructor of the simulation configuration, delete all dynamically allocated members
 */

void mcx_clearcfg(Config *cfg){
     if(cfg->medianum)
     	free(cfg->prop);
     if(cfg->detnum)
     	free(cfg->detpos);
     if(cfg->dim.x && cfg->dim.y && cfg->dim.z)
        free(cfg->vol);
     if(cfg->srcpattern)
     	free(cfg->srcpattern);
     if(cfg->replay.weight)
        free(cfg->replay.weight);
     if(cfg->replay.seed)
        free(cfg->replay.seed);
     if(cfg->replay.tof)
        free(cfg->replay.tof);
     if(cfg->replay.detid)
        free(cfg->replay.detid);
     if(cfg->dx)
        free(cfg->dx);
     if(cfg->dy)
        free(cfg->dy);
     if(cfg->dz)
        free(cfg->dz);
     if(cfg->exportfield)
        free(cfg->exportfield);
     if(cfg->exportdetected)
        free(cfg->exportdetected);
     if(cfg->exportdebugdata)
        free(cfg->exportdebugdata);
     if(cfg->seeddata)
        free(cfg->seeddata);

     mcx_initcfg(cfg);
}


/**
 * @brief Save volumetric output (fluence etc) to an Nifty format binary file
 *
 * @param[in] dat: volumetric data to be saved
 * @param[in] len: total byte length of the data to be saved
 * @param[in] name: output file name (will append '.nii')
 * @param[in] type32bit: type of the data, only support 32bit per record
 * @param[in] outputformatid: decide if save as nii or analyze format
 * @param[in] cfg: simulation configuration
 */

void mcx_savenii(float *dat, size_t len, char* name, int type32bit, int outputformatid, Config *cfg){
     FILE *fp;
     char fname[MAX_PATH_LENGTH]={'\0'};
     nifti_1_header hdr;
     nifti1_extender pad={{0,0,0,0}};
     float *logval=dat;
     size_t i;

     memset((void *)&hdr, 0, sizeof(hdr));
     hdr.sizeof_hdr = MIN_HEADER_SIZE;
     hdr.dim[0] = 4;
     hdr.dim[1] = cfg->dim.x;
     hdr.dim[2] = cfg->dim.y;
     hdr.dim[3] = cfg->dim.z;
     hdr.dim[4] = len/(cfg->dim.x*cfg->dim.y*cfg->dim.z);
     hdr.datatype = type32bit;
     hdr.bitpix = 32;
     hdr.pixdim[1] = cfg->unitinmm;
     hdr.pixdim[2] = cfg->unitinmm;
     hdr.pixdim[3] = cfg->unitinmm;
     hdr.intent_code=NIFTI_INTENT_NONE;

     if(type32bit==NIFTI_TYPE_FLOAT32){
         hdr.pixdim[4] = cfg->tstep*1e6f;
     }else{
         short *mask=(short*)logval;
	 for(i=0;i<len;i++){
	    mask[i]    =(((unsigned int *)dat)[i] & MED_MASK);
	    mask[i+len]=(((unsigned int *)dat)[i] & DET_MASK)>>16;
	 }
	 hdr.datatype = NIFTI_TYPE_UINT16;
	 hdr.bitpix = 16;
         hdr.dim[4] = 2;
         hdr.pixdim[4] = 1.f;
     }
     if (outputformatid==ofNifti){
	strncpy(hdr.magic, "n+1\0", 4);
	hdr.vox_offset = (float) NII_HEADER_SIZE;
     }else{
	strncpy(hdr.magic, "ni1\0", 4);
	hdr.vox_offset = (float)0;
     }
     hdr.scl_slope = 0.f;
     hdr.xyzt_units = NIFTI_UNITS_MM | NIFTI_UNITS_USEC;

     sprintf(fname,"%s.%s",name,outputformat[outputformatid]);

     if (( fp = fopen(fname,"wb")) == NULL)
             mcx_error(-9, "Error opening header file for write",__FILE__,__LINE__);

     if (fwrite(&hdr, MIN_HEADER_SIZE, 1, fp) != 1)
             mcx_error(-9, "Error writing header file",__FILE__,__LINE__);

     if (outputformatid==ofNifti) {
         if (fwrite(&pad, 4, 1, fp) != 1)
             mcx_error(-9, "Error writing header file extension pad",__FILE__,__LINE__);

         if (fwrite(logval, (size_t)(hdr.bitpix>>3), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp) !=
	          hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4])
             mcx_error(-9, "Error writing data to file",__FILE__,__LINE__);
	 fclose(fp);
     }else if(outputformatid==ofAnalyze){
         fclose(fp);  /* close .hdr file */

         sprintf(fname,"%s.img",name);

         fp = fopen(fname,"wb");
         if (fp == NULL)
             mcx_error(-9, "Error opening img file for write",__FILE__,__LINE__);
         if (fwrite(logval, (size_t)(hdr.bitpix>>3), hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4], fp) != 
	       hdr.dim[1]*hdr.dim[2]*hdr.dim[3]*hdr.dim[4])
             mcx_error(-9, "Error writing img file",__FILE__,__LINE__);

         fclose(fp);
     }else
         mcx_error(-9, "Output format is not supported",__FILE__,__LINE__);
}

/**
 * @brief Save volumetric output (fluence etc) to mc2 format binary file
 *
 * @param[in] dat: volumetric data to be saved
 * @param[in] len: total byte length of the data to be saved
 * @param[in] cfg: simulation configuration
 */

void mcx_savedata(float *dat, size_t len, Config *cfg){
     FILE *fp;
     char name[MAX_PATH_LENGTH];
     char fname[MAX_PATH_LENGTH];
     unsigned int glformat=GL_RGBA32F;

     if(cfg->rootpath[0])
         sprintf(name,"%s%c%s",cfg->rootpath,pathsep,cfg->session);
     else
         sprintf(name,"%s",cfg->session);

     if(cfg->outputformat==ofNifti || cfg->outputformat==ofAnalyze){
         mcx_savenii(dat, len, name, NIFTI_TYPE_FLOAT32, cfg->outputformat, cfg);
         return;
     }
     sprintf(fname,"%s.%s",name,outputformat[(int)cfg->outputformat]);
     fp=fopen(fname,"wb");

     if(fp==NULL){
	mcx_error(-2,"can not save data to disk",__FILE__,__LINE__);
     }
     if(cfg->outputformat==ofTX3){
	fwrite(&glformat,sizeof(unsigned int),1,fp);
	fwrite(&(cfg->dim.x),sizeof(int),3,fp);
     }
     fwrite(dat,sizeof(float),len,fp);
     fclose(fp);
}

/**
 * @brief Save detected photon data to mch format binary file
 *
 * @param[in] ppath: buffer pointing to the detected photon data (partial path etc)
 * @param[in] seeds: buffer pointing to the detected photon seed data
 * @param[in] count: number of detected photons
 * @param[in] doappend: flag if the new data is appended or write from the begining
 * @param[in] cfg: simulation configuration
 */

void mcx_savedetphoton(float *ppath, void *seeds, int count, int doappend, Config *cfg){
	FILE *fp;
	char fhistory[MAX_PATH_LENGTH], filetag;
	filetag=((cfg->his.detected==0  && cfg->his.savedphoton) ? 't' : 'h');
        if(cfg->rootpath[0])
                sprintf(fhistory,"%s%c%s.mc%c",cfg->rootpath,pathsep,cfg->session,filetag);
        else
                sprintf(fhistory,"%s.mc%c",cfg->session,filetag);
	if(doappend){
           fp=fopen(fhistory,"ab");
	}else{
           fp=fopen(fhistory,"wb");
	}
	if(fp==NULL){
	   mcx_error(-2,"can not save data to disk",__FILE__,__LINE__);
        }
	fwrite(&(cfg->his),sizeof(History),1,fp);
	fwrite(ppath,sizeof(float),count*cfg->his.colcount,fp);
	if(cfg->issaveseed && seeds!=NULL)
           fwrite(seeds,cfg->his.seedbyte,count,fp);
	fclose(fp);
}

/**
 * @brief Print a message to the console or a log file
 *
 * @param[in] cfg: simulation configuration
 * @param[in] str: a string to be printed
 */

void mcx_printlog(Config *cfg, char *str){
     if(cfg->flog>0){ /*stdout is 1*/
         MCX_FPRINTF(cfg->flog,"%s\n",str);
     }
}

/**
 * @brief Normalize the solution by multiplying a scaling factor
 *
 * @param[in,out] field: volumetric data before normalization
 * @param[in] scale: the scaling factor (or normalization factor) to be applied
 * @param[in] fieldlen: the length (floating point) of elements in the volume
 * @param[in] option: if set to 2, only normalize positive values (negative values for diffuse reflectance calculations)
 */

void mcx_normalize(float field[], float scale, int fieldlen, int option, int pidx, int srcnum){
     int i;
     for(i=0;i<fieldlen;i++){
         if(option==2 && field[i*srcnum+pidx]<0.f)
	     continue;
         field[i*srcnum+pidx]*=scale;
     }
}

/**
 * @brief Kahan summation: Add a sequence of finite precision floating point numbers  
 *
 * Source: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 * @param[in,out] sum: sum of the squence before and after adding the next element
 * @param[in,out] kahanc: a running compensation for lost low-order bits
 * @param[in] input: the next element of the sequence
 */
 
 void mcx_kahanSum(float *sum, float *kahanc, float input){
     float kahany=input-*kahanc;
     float kahant=*sum+kahany;
     *kahanc=kahant-*sum-kahany;
     *sum=kahant;
 }

 /**
 * @brief Retrieve mua for different cfg.vol formats to convert fluence back to energy in post-processing 
 *
 * @param[out] output: medium absorption coefficient for the current voxel
 * @param[in] mediaid: medium index of the current voxel
 * @param[in] cfg: simulation configuration
 */
 
 float mcx_updatemua(unsigned int mediaid, Config *cfg){
     float mua;
     if(cfg->mediabyte<=4)
         mua=cfg->prop[mediaid & MED_MASK].mua;
     else if(cfg->mediabyte==MEDIA_MUA_FLOAT)
         mua=fabs(*((float *)&mediaid));
     else if(cfg->mediabyte==MEDIA_ASGN_BYTE){
         union {
            unsigned i;
            unsigned char h[4];
        } val;
        val.i=mediaid & MED_MASK;
        mua=val.h[0]*(1.f/255.f)*(cfg->prop[2].mua-cfg->prop[1].mua)+cfg->prop[1].mua;
     }else if(cfg->mediabyte==MEDIA_AS_SHORT){
         union {
            unsigned int i;
            unsigned short h[2];
         } val;
        val.i=mediaid & MED_MASK;
        mua=val.h[0]*(1.f/65535.f)*(cfg->prop[2].mua-cfg->prop[1].mua)+cfg->prop[1].mua;
     }
     return mua;
 }

/**
 * @brief Force flush the command line to print the message
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_flush(Config *cfg){
#ifdef MCX_CONTAINER
    mcx_matlab_flush();
#else
    fflush(cfg->flog);
#endif
}

/**
 * @brief Error reporting function
 *
 * @param[in] id: a single integer for the types of the error
 * @param[in] msg: the error message string
 * @param[in] file: the unit file name where this error is raised
 * @param[in] linenum: the line number in the file where this error is raised
 */

void mcx_error(const int id,const char *msg,const char *file,const int linenum){
#ifdef MCX_CONTAINER
     mcx_throw_exception(id,msg,file,linenum);
#else
     MCX_FPRINTF(stdout,S_RED"\nMCX ERROR(%d):%s in unit %s:%d\n"S_RESET,id,msg,file,linenum);
     if(id==-CUDA_ERROR_LAUNCH_FAILED){
         MCX_FPRINTF(stdout,S_RED"MCX is terminated by your graphics driver. If you use windows, \n\
please modify TdrDelay value in the registry. Please checkout FAQ #1 for more details:\n\
URL: http://mcx.space/wiki/index.cgi?Doc/FAQ\n"S_RESET);
     }
     exit(id);
#endif
}

/**
 * @brief Function to recursively create output folder
 *
 * Source: https://stackoverflow.com/questions/2336242/recursive-mkdir-system-call-on-unix
 * @param[in] dir_path: folder name to be created
 * @param[in] mode: mode of the created folder
 */

int mkpath(char* dir_path, int mode){
    char* p=dir_path;
    p[strlen(p)+1]='\0';
    p[strlen(p)]=pathsep;
    for (p=strchr(dir_path+1, pathsep); p; p=strchr(p+1, pathsep)) {
      *p='\0';
      if (mkdir(dir_path, mode)==-1) {
          if (errno!=EEXIST) { *p=pathsep; return -1; }
      }
      *p=pathsep;
    }
    if(dir_path[strlen(p)-1]==pathsep)
        dir_path[strlen(p)-1]='\0';
    return 0;
}

/**
 * @brief Function to raise a CUDA error
 *
 * @param[in] ret: CUDA function return value, non-zero means an error
 */

void mcx_assert(int ret){
     if(!ret) mcx_error(ret,"assert error",__FILE__,__LINE__);
}

/**
 * @brief Read simulation settings from a configuration file (.inp or .json)
 *
 * @param[in] fname: the name of the input file (.inp or .json)
 * @param[in] cfg: simulation configuration
 */

void mcx_readconfig(char *fname, Config *cfg){
     if(fname[0]==0){
     	mcx_loadconfig(stdin,cfg);
     }else{
        FILE *fp=fopen(fname,"rt");
        if(fp==NULL && fname[0]!='{') mcx_error(-2,"can not load the specified config file",__FILE__,__LINE__);
        if(strstr(fname,".json")!=NULL || fname[0]=='{'){
            char *jbuf;
            int len;
            cJSON *jroot;

            if(fp!=NULL){
                fclose(fp);
                fp=fopen(fname,"rb");
                fseek (fp, 0, SEEK_END);
                len=ftell(fp)+1;
                jbuf=(char *)malloc(len);
                rewind(fp);
                if(fread(jbuf,len-1,1,fp)!=1)
                    mcx_error(-2,"reading input file is terminated",__FILE__,__LINE__);
                jbuf[len-1]='\0';
            }else
		jbuf=fname;
            jroot = cJSON_Parse(jbuf);
            if(jroot){
                mcx_loadjson(jroot,cfg);
                cJSON_Delete(jroot);
            }else{
                char *ptrold, *ptr=(char*)cJSON_GetErrorPtr();
                if(ptr) ptrold=strstr(jbuf,ptr);
                if(fp!=NULL) fclose(fp);
                if(ptr && ptrold){
                   char *offs=(ptrold-jbuf>=50) ? ptrold-50 : jbuf;
                   while(offs<ptrold){
                      MCX_FPRINTF(stderr,"%c",*offs);
                      offs++;
                   }
                   MCX_FPRINTF(stderr,"<error>%.50s\n",ptrold);
                }
                if(fp!=NULL) free(jbuf);
                mcx_error(-9,"invalid JSON input file",__FILE__,__LINE__);
            }
            if(fp!=NULL) free(jbuf);
        }else{
	    mcx_loadconfig(fp,cfg); 
        }
        if(fp!=NULL) fclose(fp);
	if(cfg->session[0]=='\0'){
	    strncpy(cfg->session,fname,MAX_SESSION_LENGTH);
	}
     }
     if(cfg->rootpath[0]!='\0'){
	struct stat st = {0};
	if (stat((const char *)cfg->rootpath, &st) == -1) {
	    if(mkpath(cfg->rootpath, 0755))
	       mcx_error(-9,"can not create output folder",__FILE__,__LINE__);
	}
     }
}

/**
 * @brief Write simulation settings to an inp file
 *
 * @param[in] fname: the name of the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_writeconfig(char *fname, Config *cfg){
     if(fname[0]==0)
     	mcx_saveconfig(stdout,cfg);
     else{
     	FILE *fp=fopen(fname,"wt");
	if(fp==NULL) mcx_error(-2,"can not write to the specified config file",__FILE__,__LINE__);
	mcx_saveconfig(fp,cfg);     
	fclose(fp);
     }
}

/**
 * @brief Preprocess user input and prepare the volumetric domain for simulation
 *
 * This function preprocess the user input and prepare the domain for the simulation.
 * It loads the media index array from file, add detector masks for easy detection, and
 * check inconsistency between the user specified inputs.
 *
 * @param[in] filename: the name of the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_prepdomain(char *filename, Config *cfg){
     if(filename[0] || cfg->vol){
        if(cfg->vol==NULL){
	     mcx_loadvolume(filename,cfg);
	     if(cfg->shapedata && strstr(cfg->shapedata,":")!=NULL){
	          int status;
     		  Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
        	  if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
		  status=mcx_parse_shapestring(&grid,cfg->shapedata);
		  if(status){
		      MCX_ERROR(status,mcx_last_shapeerror());
		  }
	     }
	}
	if(cfg->isrowmajor){
		/*from here on, the array is always col-major*/
		mcx_convertrow2col(&(cfg->vol), &(cfg->dim));
		cfg->isrowmajor=0;
	}
	if(cfg->issavedet)
		mcx_maskdet(cfg);
	if(cfg->isdumpmask)
	        mcx_dumpmask(cfg);
     }else{
     	mcx_error(-4,"one must specify a binary volume file in order to run the simulation",__FILE__,__LINE__);
     }
     if(cfg->respin==0)
        mcx_error(-1,"respin number can not be 0, check your -r/--repeat input or cfg.respin value",__FILE__,__LINE__);

     if(cfg->seed==SEED_FROM_FILE && cfg->seedfile[0]){
        if(cfg->respin>1 || cfg->respin<0){
	   cfg->respin=1;
	   fprintf(stderr,S_RED"WARNING: respin is disabled in the replay mode\n"S_RESET);
	}
        mcx_loadseedfile(cfg);
     }
     if(cfg->replaydet>(int)cfg->detnum)
        MCX_ERROR(-4,"replay detector ID exceeds the maximum detector number");
     if(cfg->replaydet==-1 && cfg->detnum==1)
        cfg->replaydet=1;
     if(cfg->medianum){
        for(int i=0;i<cfg->medianum;i++){
             if(cfg->prop[i].mus==0.f){
	         cfg->prop[i].mus=EPS;
		 cfg->prop[i].g=1.f;
	     }
	}
     }
     for(int i=0;i<MAX_DEVICE;i++)
        if(cfg->deviceid[i]=='0')
           cfg->deviceid[i]='\0';

     for(int i=0;i<6;i++)
        if(cfg->bc[i]>='A' && mcx_lookupindex(cfg->bc+i,boundarycond))
	   MCX_ERROR(-4,"unknown boundary condition specifier");

     if((cfg->mediabyte==MEDIA_AS_F2H || cfg->mediabyte==MEDIA_MUA_FLOAT || cfg->mediabyte==MEDIA_AS_HALF) && cfg->medianum<2)
         MCX_ERROR(-4,"the 'prop' field must contain at least 2 rows for the requested media format");
     if((cfg->mediabyte==MEDIA_ASGN_BYTE || cfg->mediabyte==MEDIA_AS_SHORT) && cfg->medianum<3)
         MCX_ERROR(-4,"the 'prop' field must contain at least 3 rows for the requested media format");

     if(cfg->ismomentum)
         cfg->savedetflag=SET_SAVE_MOM(cfg->savedetflag);
     if(cfg->issaveexit){
         cfg->savedetflag=SET_SAVE_PEXIT(cfg->savedetflag);
	 cfg->savedetflag=SET_SAVE_VEXIT(cfg->savedetflag);
     }
     if(cfg->issavedet && cfg->savedetflag==0)
         cfg->savedetflag=0x5;
     if(cfg->mediabyte>=100){
	 cfg->savedetflag=UNSET_SAVE_NSCAT(cfg->savedetflag);
	 cfg->savedetflag=UNSET_SAVE_PPATH(cfg->savedetflag);
	 cfg->savedetflag=UNSET_SAVE_MOM(cfg->savedetflag);
     }
     if(cfg->issaveref>1){
        if(cfg->issavedet==0)
	    MCX_ERROR(-4,"you must have at least two outputs if issaveref is greater than 1");

        if(cfg->dim.x*cfg->dim.y*cfg->dim.z > cfg->maxdetphoton){
	    MCX_FPRINTF(cfg->flog,"you must set --maxdetphoton larger than the total size of the voxels when --issaveref is greater than 1; autocorrecting ...\n");
	    cfg->maxdetphoton=cfg->dim.x*cfg->dim.y*cfg->dim.z;
        }
	cfg->savedetflag=0x5;
     }
}

/**
 * @brief Load user inputs from a .inp input file
 *
 * This function loads user input from a simple text input format in a .inp extension
 *
 * @param[in] in: file handle to the .inp file
 * @param[in] cfg: simulation configuration
 */

void mcx_loadconfig(FILE *in, Config *cfg){
     uint i,gates,itmp;
     size_t count;
     float dtmp;
     char filename[MAX_PATH_LENGTH]={'\0'}, comment[MAX_PATH_LENGTH],strtypestr[MAX_SESSION_LENGTH]={'\0'},*comm;
     
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of photons: [1000000]\n\t");
     MCX_ASSERT(fscanf(in,"%ld", &(count) )==1);
     if(cfg->nphoton==0) cfg->nphoton=count;
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%ld\nPlease specify the random number generator seed: [1234567]\n\t",cfg->nphoton);
     if(cfg->seed==0){
        MCX_ASSERT(fscanf(in,"%d", &(cfg->seed) )==1);
     }else{
        MCX_ASSERT(fscanf(in,"%d", &itmp )==1);
     }
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d\nPlease specify the position of the source (in grid unit): [10 10 5]\n\t",cfg->seed);
     MCX_ASSERT(fscanf(in,"%f %f %f", &(cfg->srcpos.x),&(cfg->srcpos.y),&(cfg->srcpos.z) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(cfg->issrcfrom0==0 && comm!=NULL && sscanf(comm,"%d",&itmp)==1)
         cfg->issrcfrom0=itmp;

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the normal direction of the source fiber: [0 0 1]\n\t",
                                   cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z);
     if(!cfg->issrcfrom0){
        cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
     }
     MCX_ASSERT(fscanf(in,"%f %f %f", &(cfg->srcdir.x),&(cfg->srcdir.y),&(cfg->srcdir.z) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(comm!=NULL && sscanf(comm,"%f",&dtmp)==1)
         cfg->srcdir.w=dtmp;

     if(in==stdin)
     	fprintf(stdout,"%f %f %f %f\nPlease specify the time gates (format: start end step) in seconds [0.0 1e-9 1e-10]\n\t",
                                   cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z,cfg->srcdir.w);
     MCX_ASSERT(fscanf(in,"%f %f %f", &(cfg->tstart),&(cfg->tend),&(cfg->tstep) )==3);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the path to the volume binary file:\n\t",
                                   cfg->tstart,cfg->tend,cfg->tstep);
     if(cfg->tstart>cfg->tend || cfg->tstep==0.f){
         mcx_error(-9,"incorrect time gate settings",__FILE__,__LINE__);
     }
     gates=(uint)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
     if(cfg->maxgate==0)
	 cfg->maxgate=gates;
     else if(cfg->maxgate>gates)
	 cfg->maxgate=gates;

     MCX_ASSERT(fscanf(in,"%s", filename)==1);
     if(cfg->rootpath[0]){
#ifdef WIN32
         sprintf(comment,"%s\\%s",cfg->rootpath,filename);
#else
         sprintf(comment,"%s/%s",cfg->rootpath,filename);
#endif
         strncpy(filename,comment,MAX_PATH_LENGTH);
     }
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%s\nPlease specify the x voxel size (in mm), x dimension, min and max x-index [1.0 100 1 100]:\n\t",filename);
     MCX_ASSERT(fscanf(in,"%f %d %d %d", &(cfg->steps.x),&(cfg->dim.x),&(cfg->crop0.x),&(cfg->crop1.x))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the y voxel size (in mm), y dimension, min and max y-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.x,cfg->dim.x,cfg->crop0.x,cfg->crop1.x);
     MCX_ASSERT(fscanf(in,"%f %d %d %d", &(cfg->steps.y),&(cfg->dim.y),&(cfg->crop0.y),&(cfg->crop1.y))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the z voxel size (in mm), z dimension, min and max z-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.y,cfg->dim.y,cfg->crop0.y,cfg->crop1.y);
     MCX_ASSERT(fscanf(in,"%f %d %d %d", &(cfg->steps.z),&(cfg->dim.z),&(cfg->crop0.z),&(cfg->crop1.z))==4);
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(cfg->steps.x!=cfg->steps.y || cfg->steps.y!=cfg->steps.z)
        mcx_error(-9,"MCX currently does not support anisotropic voxels",__FILE__,__LINE__);

     if(cfg->steps.x!=1.f && cfg->unitinmm==1.f)
        cfg->unitinmm=cfg->steps.x;

     if(cfg->unitinmm!=1.f){
        cfg->steps.x=cfg->unitinmm; cfg->steps.y=cfg->unitinmm; cfg->steps.z=cfg->unitinmm;
     }

     if(cfg->sradius>0.f){
     	cfg->crop0.x=MAX((uint)(cfg->srcpos.x-cfg->sradius),0);
     	cfg->crop0.y=MAX((uint)(cfg->srcpos.y-cfg->sradius),0);
     	cfg->crop0.z=MAX((uint)(cfg->srcpos.z-cfg->sradius),0);
     	cfg->crop1.x=MIN((uint)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
     	cfg->crop1.y=MIN((uint)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
     	cfg->crop1.z=MIN((uint)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
     }else if(cfg->sradius==0.f){
     	memset(&(cfg->crop0),0,sizeof(uint3));
     	memset(&(cfg->crop1),0,sizeof(uint3));
     }else{
        /*
           if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
        */
        if(!cfg->issrcfrom0){
            cfg->crop0.x--;cfg->crop0.y--;cfg->crop0.z--;  /*convert to C index*/
            cfg->crop1.x--;cfg->crop1.y--;cfg->crop1.z--;
        }
     }
     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the total types of media:\n\t",
                                  cfg->steps.z,cfg->dim.z,cfg->crop0.z,cfg->crop1.z);
     MCX_ASSERT(fscanf(in,"%d", &(cfg->medianum))==1);
     cfg->medianum++;
     comm=fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%d\n",cfg->medianum);
     cfg->prop=(Medium*)malloc(sizeof(Medium)*cfg->medianum);
     cfg->prop[0].mua=0.f; /*property 0 is already air*/
     cfg->prop[0].mus=0.f;
     cfg->prop[0].g=1.f;
     cfg->prop[0].n=1.f;
     for(i=1;i<cfg->medianum;i++){
        if(in==stdin)
		fprintf(stdout,"Please define medium #%d: mus(1/mm), anisotropy, mua(1/mm) and refractive index: [1.01 0.01 0.04 1.37]\n\t",i);
     	MCX_ASSERT(fscanf(in, "%f %f %f %f", &(cfg->prop[i].mus),&(cfg->prop[i].g),&(cfg->prop[i].mua),&(cfg->prop[i].n))==4);
        comm=fgets(comment,MAX_PATH_LENGTH,in);
        if(in==stdin)
		fprintf(stdout,"%f %f %f %f\n",cfg->prop[i].mus,cfg->prop[i].g,cfg->prop[i].mua,cfg->prop[i].n);
     }
     if(cfg->unitinmm!=1.f){
         for(i=1;i<cfg->medianum;i++){
		cfg->prop[i].mus*=cfg->unitinmm;
		cfg->prop[i].mua*=cfg->unitinmm;
         }
     }
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of detectors and fiber diameter (in grid unit):\n\t");
     MCX_ASSERT(fscanf(in,"%d %f", &(cfg->detnum), &(cfg->detradius))==2);
     comm=fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d %f\n",cfg->detnum,cfg->detradius);
     if(cfg->medianum+cfg->detnum>MAX_PROP_AND_DETECTORS)
         mcx_error(-4,"input media types plus detector number exceeds the maximum total (4000)",__FILE__,__LINE__);

     cfg->detpos=(float4*)malloc(sizeof(float4)*cfg->detnum);
     if(cfg->issavedet && cfg->detnum==0) 
      	cfg->issavedet=0;
     for(i=0;i<cfg->detnum;i++){
        if(in==stdin)
		fprintf(stdout,"Please define detector #%d: x,y,z (in grid unit): [5 5 5 1]\n\t",i);
     	MCX_ASSERT(fscanf(in, "%f %f %f", &(cfg->detpos[i].x),&(cfg->detpos[i].y),&(cfg->detpos[i].z))==3);
	cfg->detpos[i].w=cfg->detradius;
        if(!cfg->issrcfrom0){
		cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
	}
        comm=fgets(comment,MAX_PATH_LENGTH,in);
        if(comm!=NULL && sscanf(comm,"%f",&dtmp)==1)
            cfg->detpos[i].w=dtmp;

        if(in==stdin)
		fprintf(stdout,"%f %f %f\n",cfg->detpos[i].x,cfg->detpos[i].y,cfg->detpos[i].z);
     }
     mcx_prepdomain(filename,cfg);
     cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.srcnum=cfg->srcnum;
     cfg->his.savedetflag=cfg->savedetflag;
     //cfg->his.colcount=2+(cfg->medianum-1)*(2+(cfg->ismomentum>0))+(cfg->issaveexit>0)*6; /*column count=maxmedia+2*/

     if(in==stdin)
     	fprintf(stdout,"Please specify the source type[pencil|cone|gaussian]:\n\t");
     if(fscanf(in,"%s", strtypestr)==1 && strtypestr[0]){
        int srctype=mcx_keylookup(strtypestr,srctypeid);
	if(srctype==-1)
	   MCX_ERROR(-6,"the specified source type is not supported");
        if(srctype>=0){
           comm=fgets(comment,MAX_PATH_LENGTH,in);
	   cfg->srctype=srctype;
	   if(in==stdin)
     	      fprintf(stdout,"Please specify the source parameters set 1 (4 floating-points):\n\t");
           mcx_assert(fscanf(in, "%f %f %f %f", &(cfg->srcparam1.x),
	          &(cfg->srcparam1.y),&(cfg->srcparam1.z),&(cfg->srcparam1.w))==4);
	   if(in==stdin)
     	      fprintf(stdout,"Please specify the source parameters set 2 (4 floating-points):\n\t");
           mcx_assert(fscanf(in, "%f %f %f %f", &(cfg->srcparam2.x),
	          &(cfg->srcparam2.y),&(cfg->srcparam2.z),&(cfg->srcparam2.w))==4);
           if(cfg->srctype==MCX_SRC_PATTERN && cfg->srcparam1.w*cfg->srcparam2.w>0){
               char patternfile[MAX_PATH_LENGTH];
               FILE *fp;
               if(cfg->srcpattern) free(cfg->srcpattern);
               cfg->srcpattern=(float*)calloc((cfg->srcparam1.w*cfg->srcparam2.w*cfg->srcnum),sizeof(float));
               MCX_ASSERT(fscanf(in, "%s", patternfile)==1);
               fp=fopen(patternfile,"rb");
               if(fp==NULL)
                     MCX_ERROR(-6,"pattern file can not be opened");
               MCX_ASSERT(fread(cfg->srcpattern,cfg->srcparam1.w*cfg->srcparam2.w*cfg->srcnum,sizeof(float),fp)==sizeof(float));
               fclose(fp);
           }else if(cfg->srctype==MCX_SRC_PATTERN3D && cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z>0){
               char patternfile[MAX_PATH_LENGTH];
               FILE *fp;
               if(cfg->srcpattern) free(cfg->srcpattern);
               cfg->srcpattern=(float*)calloc((int)(cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z*cfg->srcnum),sizeof(float));
               MCX_ASSERT(fscanf(in, "%s", patternfile)==1);
               fp=fopen(patternfile,"rb");
               if(fp==NULL)
                     MCX_ERROR(-6,"pattern file can not be opened");
               MCX_ASSERT(fread(cfg->srcpattern,cfg->srcparam1.x*cfg->srcparam1.y*cfg->srcparam1.z*cfg->srcnum,sizeof(float),fp)==sizeof(float));
               fclose(fp);
           }
	}else
	   return;
     }else
        return;
}

/**
 * @brief Load user inputs from a .json input file
 *
 * This function loads user input from a JSON format in a .json extension
 *
 * @param[out] root: json data structure pointer
 * @param[in] cfg: simulation configuration
 */

int mcx_loadjson(cJSON *root, Config *cfg){
     int i;
     cJSON *Domain, *Optode, *Forward, *Session, *Shapes, *tmp, *subitem;
     char filename[MAX_PATH_LENGTH]={'\0'};
     Domain  = cJSON_GetObjectItem(root,"Domain");
     Optode  = cJSON_GetObjectItem(root,"Optode");
     Session = cJSON_GetObjectItem(root,"Session");
     Forward = cJSON_GetObjectItem(root,"Forward");
     Shapes  = cJSON_GetObjectItem(root,"Shapes");

     if(Domain){
        char volfile[MAX_PATH_LENGTH];
	cJSON *meds,*val,*vv;
	val=FIND_JSON_OBJ("VolumeFile","Domain.VolumeFile",Domain);
	if(val){
          strncpy(volfile, val->valuestring, MAX_PATH_LENGTH);
          if(cfg->rootpath[0]){
#ifdef WIN32
           sprintf(filename,"%s\\%s",cfg->rootpath,volfile);
#else
           sprintf(filename,"%s/%s",cfg->rootpath,volfile);
#endif
          }else{
	     strncpy(filename,volfile,MAX_PATH_LENGTH);
	  }
	}
	val=FIND_JSON_OBJ("MediaFormat","Domain.MediaFormat",Domain);
	if(val){
            cfg->mediabyte=mcx_keylookup((char*)FIND_JSON_KEY("MediaFormat","Domain.MediaFormat",Domain,"byte",valuestring),mediaformat);
	    if(cfg->mediabyte==-1)
	       MCX_ERROR(-1,"Unsupported media format.");
	    cfg->mediabyte=mediaformatid[cfg->mediabyte];
	}
        if(!flagset['u'])
	    cfg->unitinmm=FIND_JSON_KEY("LengthUnit","Domain.LengthUnit",Domain,1.f,valuedouble);
        meds=FIND_JSON_OBJ("Media","Domain.Media",Domain);
        if(meds){
           cJSON *med=meds->child;
           if(med){
             cfg->medianum=cJSON_GetArraySize(meds);
             cfg->prop=(Medium*)malloc(sizeof(Medium)*cfg->medianum);
             for(i=0;i<cfg->medianum;i++){
               cJSON *val=FIND_JSON_OBJ("mua",(MCX_ERROR(-1,"You must specify absorption coeff, default in 1/mm"),""),med);
               if(val) cfg->prop[i].mua=val->valuedouble;
	       val=FIND_JSON_OBJ("mus",(MCX_ERROR(-1,"You must specify scattering coeff, default in 1/mm"),""),med);
               if(val) cfg->prop[i].mus=val->valuedouble;
	       val=FIND_JSON_OBJ("g",(MCX_ERROR(-1,"You must specify anisotropy [0-1]"),""),med);
               if(val) cfg->prop[i].g=val->valuedouble;
	       val=FIND_JSON_OBJ("n",(MCX_ERROR(-1,"You must specify refractive index"),""),med);
	       if(val) cfg->prop[i].n=val->valuedouble;

               med=med->next;
               if(med==NULL) break;
             }
	     if(cfg->unitinmm!=1.f){
        	 for(i=0;i<cfg->medianum;i++){
			cfg->prop[i].mus*=cfg->unitinmm;
			cfg->prop[i].mua*=cfg->unitinmm;
        	 }
	     }
           }
        }
	val=FIND_JSON_OBJ("Dim","Domain.Dim",Domain);
	if(val && cJSON_GetArraySize(val)>=3){
	   cfg->dim.x=val->child->valueint;
           cfg->dim.y=val->child->next->valueint;
           cfg->dim.z=val->child->next->next->valueint;
	}else{
	   if(!Shapes)
	      MCX_ERROR(-1,"You must specify the dimension of the volume");
	}
	val=FIND_JSON_OBJ("Step","Domain.Step",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->steps.x=val->child->valuedouble;
               cfg->steps.y=val->child->next->valuedouble;
               cfg->steps.z=val->child->next->next->valuedouble;
           }else{
	       MCX_ERROR(-1,"Domain::Step has incorrect element numbers");
           }
	}
	if(cfg->steps.x!=cfg->steps.y || cfg->steps.y!=cfg->steps.z)
           mcx_error(-9,"MCX currently does not support anisotropic voxels",__FILE__,__LINE__);

	if(cfg->steps.x!=1.f && cfg->unitinmm==1.f)
           cfg->unitinmm=cfg->steps.x;

	if(cfg->unitinmm!=1.f){
           cfg->steps.x=cfg->unitinmm; cfg->steps.y=cfg->unitinmm; cfg->steps.z=cfg->unitinmm;
	}
	val=FIND_JSON_OBJ("VoxelSize","Domain.VoxelSize",Domain);
	if(val){
	   val=FIND_JSON_OBJ("Dx","Domain.VoxelSize.Dx",Domain);
	   if(cJSON_GetArraySize(val)>=1){
	       int len=cJSON_GetArraySize(val);
	       if(len==1)
	           cfg->steps.x=-1.0f;
	       else if(len==cfg->dim.x)
	           cfg->steps.x=-2.0f;
	       else
	           MCX_ERROR(-1,"Domain::VoxelSize::Dx has incorrect element numbers");
	       cfg->dx=malloc(sizeof(float)*len);
	       vv=val->child;
	       for(i=0;i<len;i++){
	          cfg->dx[i]=vv->valuedouble;
		  vv=vv->next;
	       }
           }
	   val=FIND_JSON_OBJ("Dy","Domain.VoxelSize.Dy",Domain);
	   if(cJSON_GetArraySize(val)>=1){
	       int len=cJSON_GetArraySize(val);
	       if(len==1)
	           cfg->steps.y=-1.0f;
	       else if(len==cfg->dim.y)
	           cfg->steps.y=-2.0f;
	       else
	           MCX_ERROR(-1,"Domain::VoxelSize::Dy has incorrect element numbers");
	       cfg->dy=malloc(sizeof(float)*len);
	       vv=val->child;
	       for(i=0;i<len;i++){
	          cfg->dy[i]=vv->valuedouble;
		  vv=vv->next;
	       }
           }
	   val=FIND_JSON_OBJ("Dz","Domain.VoxelSize.Dz",Domain);
	   if(cJSON_GetArraySize(val)>=1){
	       int len=cJSON_GetArraySize(val);
	       if(len==1)
	           cfg->steps.z=-1.0f;
	       else if(len==cfg->dim.z)
	           cfg->steps.z=-2.0f;
	       else
	           MCX_ERROR(-1,"Domain::VoxelSize::Dz has incorrect element numbers");
	       cfg->dz=malloc(sizeof(float)*len);
	       vv=val->child;
	       for(i=0;i<len;i++){
	          cfg->dz[i]=vv->valuedouble;
		  vv=vv->next;
	       }
           }
	}
	val=FIND_JSON_OBJ("CacheBoxP0","Domain.CacheBoxP0",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->crop0.x=val->child->valueint;
               cfg->crop0.y=val->child->next->valueint;
               cfg->crop0.z=val->child->next->next->valueint;
           }else{
	       MCX_ERROR(-1,"Domain::CacheBoxP0 has incorrect element numbers");
           }
	}
	val=FIND_JSON_OBJ("CacheBoxP1","Domain.CacheBoxP1",Domain);
	if(val){
	   if(cJSON_GetArraySize(val)>=3){
	       cfg->crop1.x=val->child->valueint;
               cfg->crop1.y=val->child->next->valueint;
               cfg->crop1.z=val->child->next->next->valueint;
           }else{
	       MCX_ERROR(-1,"Domain::CacheBoxP1 has incorrect element numbers");
           }
	}
	val=FIND_JSON_OBJ("OriginType","Domain.OriginType",Domain);
	if(val && cfg->issrcfrom0==0) cfg->issrcfrom0=val->valueint;

	if(cfg->sradius>0.f){
     	   cfg->crop0.x=MAX((uint)(cfg->srcpos.x-cfg->sradius),0);
     	   cfg->crop0.y=MAX((uint)(cfg->srcpos.y-cfg->sradius),0);
     	   cfg->crop0.z=MAX((uint)(cfg->srcpos.z-cfg->sradius),0);
     	   cfg->crop1.x=MIN((uint)(cfg->srcpos.x+cfg->sradius),cfg->dim.x-1);
     	   cfg->crop1.y=MIN((uint)(cfg->srcpos.y+cfg->sradius),cfg->dim.y-1);
     	   cfg->crop1.z=MIN((uint)(cfg->srcpos.z+cfg->sradius),cfg->dim.z-1);
	}else if(cfg->sradius==0.f){
     	   memset(&(cfg->crop0),0,sizeof(uint3));
     	   memset(&(cfg->crop1),0,sizeof(uint3));
	}else{
           /*
              if -R is followed by a negative radius, mcx uses crop0/crop1 to set the cachebox
           */
           if(!cfg->issrcfrom0){
               cfg->crop0.x--;cfg->crop0.y--;cfg->crop0.z--;  /*convert to C index*/
               cfg->crop1.x--;cfg->crop1.y--;cfg->crop1.z--;
           }
	}
     }
     if(Optode){
        cJSON *dets, *src=FIND_JSON_OBJ("Source","Optode.Source",Optode);
        if(src){
           subitem=FIND_JSON_OBJ("Pos","Optode.Source.Pos",src);
           if(subitem){
              cfg->srcpos.x=subitem->child->valuedouble;
              cfg->srcpos.y=subitem->child->next->valuedouble;
              cfg->srcpos.z=subitem->child->next->next->valuedouble;
           }
           subitem=FIND_JSON_OBJ("Dir","Optode.Source.Dir",src);
           if(subitem){
              cfg->srcdir.x=subitem->child->valuedouble;
              cfg->srcdir.y=subitem->child->next->valuedouble;
              cfg->srcdir.z=subitem->child->next->next->valuedouble;
	      if(subitem->child->next->next->next)
	         cfg->srcdir.w=subitem->child->next->next->next->valuedouble;
           }
	   if(!cfg->issrcfrom0){
              cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index, grid center*/
	   }
           cfg->srctype=mcx_keylookup((char*)FIND_JSON_KEY("Type","Optode.Source.Type",src,"pencil",valuestring),srctypeid);
           subitem=FIND_JSON_OBJ("Param1","Optode.Source.Param1",src);
           if(subitem){
              cfg->srcparam1.x=subitem->child->valuedouble;
              if(subitem->child->next){
		      cfg->srcparam1.y=subitem->child->next->valuedouble;
                      if(subitem->child->next->next){
                          cfg->srcparam1.z=subitem->child->next->next->valuedouble;
			  if(subitem->child->next->next->next)
                              cfg->srcparam1.w=subitem->child->next->next->next->valuedouble;
		      }
	      }
           }
           subitem=FIND_JSON_OBJ("Param2","Optode.Source.Param2",src);
           if(subitem){
              cfg->srcparam2.x=subitem->child->valuedouble;
              if(subitem->child->next){
		      cfg->srcparam2.y=subitem->child->next->valuedouble;
                      if(subitem->child->next->next){
                          cfg->srcparam2.z=subitem->child->next->next->valuedouble;
			  if(subitem->child->next->next->next)
                              cfg->srcparam2.w=subitem->child->next->next->next->valuedouble;
		      }
	      }
           }
	   cfg->srcnum=FIND_JSON_KEY("SrcNum","Optode.Source.SrcNum",src,cfg->srcnum,valueint);
           subitem=FIND_JSON_OBJ("Pattern","Optode.Source.Pattern",src);
           if(subitem){
              int nx=FIND_JSON_KEY("Nx","Optode.Source.Pattern.Nx",subitem,0,valueint);
              int ny=FIND_JSON_KEY("Ny","Optode.Source.Pattern.Ny",subitem,0,valueint);
              int nz=FIND_JSON_KEY("Nz","Optode.Source.Pattern.Nz",subitem,1,valueint);
              if(nx>0 && ny>0){
                 cJSON *pat=FIND_JSON_OBJ("Data","Optode.Source.Pattern.Data",subitem);
                 if(pat && pat->child){
                     int i;
                     pat=pat->child;
                     if(cfg->srcpattern) free(cfg->srcpattern);
                     cfg->srcpattern=(float*)calloc(nx*ny*nz*cfg->srcnum,sizeof(float));
                     for(i=0;i<nx*ny*nz*cfg->srcnum;i++){
                         if(pat==NULL)
                             MCX_ERROR(-1,"Incomplete pattern data");
                         cfg->srcpattern[i]=pat->valuedouble;
                         pat=pat->next;
                     }
                 }else if(pat){
                     FILE *fid=fopen(pat->valuestring,"rb");
		     if(fid!=NULL){
		         if(cfg->srcpattern) free(cfg->srcpattern);
                         cfg->srcpattern=(float*)calloc(nx*ny*nz*cfg->srcnum,sizeof(float));
                         fread((void*)cfg->srcpattern,sizeof(float),nx*ny*nz*cfg->srcnum,fid);
			 fclose(fid);
                     }
		 }
              }
           }
        }
        dets=FIND_JSON_OBJ("Detector","Optode.Detector",Optode);
        if(dets){
           cJSON *det=dets->child;
           if(det){
             cfg->detnum=cJSON_GetArraySize(dets);
             cfg->detpos=(float4*)malloc(sizeof(float4)*cfg->detnum);
	     if(cfg->issavedet && cfg->detnum==0) 
      		cfg->issavedet=0;
             for(i=0;i<cfg->detnum;i++){
               cJSON *pos=dets, *rad=NULL;
               rad=FIND_JSON_OBJ("R","Optode.Detector.R",det);
               if(cJSON_GetArraySize(det)==2){
                   pos=FIND_JSON_OBJ("Pos","Optode.Detector.Pos",det);
               }
               if(pos){
	           cfg->detpos[i].x=pos->child->valuedouble;
                   cfg->detpos[i].y=pos->child->next->valuedouble;
	           cfg->detpos[i].z=pos->child->next->next->valuedouble;
               }
               if(rad){
                   cfg->detpos[i].w=rad->valuedouble;
               }
               if(!cfg->issrcfrom0){
		   cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
	       }
               det=det->next;
               if(det==NULL) break;
             }
           }
        }
     }
     if(cfg->medianum+cfg->detnum>MAX_PROP_AND_DETECTORS)
         mcx_error(-4,"input media types plus detector number exceeds the maximum total (4000)",__FILE__,__LINE__);
     if(Session){
        char val[1];
	if(!flagset['E'])  cfg->seed=FIND_JSON_KEY("RNGSeed","Session.RNGSeed",Session,-1,valueint);
        if(!flagset['n'])  cfg->nphoton=FIND_JSON_KEY("Photons","Session.Photons",Session,0,valuedouble);
        if(cfg->session[0]=='\0')  strncpy(cfg->session, FIND_JSON_KEY("ID","Session.ID",Session,"default",valuestring), MAX_SESSION_LENGTH);
        if(cfg->rootpath[0]=='\0') strncpy(cfg->rootpath, FIND_JSON_KEY("RootPath","Session.RootPath",Session,"",valuestring), MAX_PATH_LENGTH);

        if(!flagset['b'])  cfg->isreflect=FIND_JSON_KEY("DoMismatch","Session.DoMismatch",Session,cfg->isreflect,valueint);
        if(!flagset['S'])  cfg->issave2pt=FIND_JSON_KEY("DoSaveVolume","Session.DoSaveVolume",Session,cfg->issave2pt,valueint);
        if(!flagset['U'])  cfg->isnormalized=FIND_JSON_KEY("DoNormalize","Session.DoNormalize",Session,cfg->isnormalized,valueint);
        if(!flagset['d'])  cfg->issavedet=FIND_JSON_KEY("DoPartialPath","Session.DoPartialPath",Session,cfg->issavedet,valueint);
        if(!flagset['X'])  cfg->issaveref=FIND_JSON_KEY("DoSaveRef","Session.DoSaveRef",Session,cfg->issaveref,valueint);
        if(!flagset['x'])  cfg->issaveexit=FIND_JSON_KEY("DoSaveExit","Session.DoSaveExit",Session,cfg->issaveexit,valueint);
        if(!flagset['q'])  cfg->issaveseed=FIND_JSON_KEY("DoSaveSeed","Session.DoSaveSeed",Session,cfg->issaveseed,valueint);
        if(!flagset['A'])  cfg->autopilot=FIND_JSON_KEY("DoAutoThread","Session.DoAutoThread",Session,cfg->autopilot,valueint);
	if(!flagset['m'])  cfg->ismomentum=FIND_JSON_KEY("DoDCS","Session.DoDCS",Session,cfg->ismomentum,valueint);
	if(!flagset['V'])  cfg->isspecular=FIND_JSON_KEY("DoSpecular","Session.DoSpecular",Session,cfg->isspecular,valueint);
	if(!flagset['D'])  cfg->debuglevel=mcx_parsedebugopt(FIND_JSON_KEY("DebugFlag","Session.DebugFlag",Session,"",valuestring),debugflag);
	if(!flagset['w'])  cfg->savedetflag=mcx_parsedebugopt(FIND_JSON_KEY("SaveDataMask","Session.SaveDataMask",Session,"",valuestring),saveflag);

        if(!cfg->outputformat)  cfg->outputformat=mcx_keylookup((char *)FIND_JSON_KEY("OutputFormat","Session.OutputFormat",Session,"mc2",valuestring),outputformat);
        if(cfg->outputformat<0)
                mcx_error(-2,"the specified output format is not recognized",__FILE__,__LINE__);

        strncpy(val,FIND_JSON_KEY("OutputType","Session.OutputType",Session,outputtype+cfg->outputtype,valuestring),1);
        if(mcx_lookupindex(val, outputtype)){
                mcx_error(-2,"the specified output data type is not recognized",__FILE__,__LINE__);
        }
	if(!flagset['O']) cfg->outputtype=val[0];
     }
     if(Forward){
        uint gates;
        cfg->tstart=FIND_JSON_KEY("T0","Forward.T0",Forward,0.0,valuedouble);
        cfg->tend  =FIND_JSON_KEY("T1","Forward.T1",Forward,0.0,valuedouble);
        cfg->tstep =FIND_JSON_KEY("Dt","Forward.Dt",Forward,0.0,valuedouble);
	if(cfg->tstart>cfg->tend || cfg->tstep==0.f)
            mcx_error(-9,"incorrect time gate settings",__FILE__,__LINE__);

        gates=(uint)((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
        if(cfg->maxgate==0)
            cfg->maxgate=gates;
        else if(cfg->maxgate>gates)
            cfg->maxgate=gates;
     }
     if(filename[0]=='\0'){
         if(Shapes){
             int status;
             Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
             if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
	     status=mcx_parse_jsonshapes(root, &grid);
	     if(status){
	         MCX_ERROR(status,mcx_last_shapeerror());
	     }
	 }else{
	     MCX_ERROR(-1,"You must either define Domain.VolumeFile, or define a Shapes section");
	 }
     }else if(Shapes){
         MCX_ERROR(-1,"You can not specify both Domain.VolumeFile and Shapes sections");
     }
     mcx_prepdomain(filename,cfg);
     cfg->his.maxmedia=cfg->medianum-1; /*skip media 0*/
     cfg->his.detnum=cfg->detnum;
     cfg->his.srcnum=cfg->srcnum;
     cfg->his.savedetflag=cfg->savedetflag;
     //cfg->his.colcount=2+(cfg->medianum-1)*(2+(cfg->ismomentum>0))+(cfg->issaveexit>0)*6; /*column count=maxmedia+2*/
     return 0;
}

/**
 * @brief Save simulation settings to an inp file
 *
 * @param[in] out: handle to the output file
 * @param[in] cfg: simulation configuration
 */

void mcx_saveconfig(FILE *out, Config *cfg){
     uint i;

     fprintf(out,"%ld\n", (cfg->nphoton) );
     fprintf(out,"%d\n", (cfg->seed) );
     fprintf(out,"%f %f %f\n", (cfg->srcpos.x),(cfg->srcpos.y),(cfg->srcpos.z) );
     fprintf(out,"%f %f %f\n", (cfg->srcdir.x),(cfg->srcdir.y),(cfg->srcdir.z) );
     fprintf(out,"%e %e %e\n", (cfg->tstart),(cfg->tend),(cfg->tstep) );
     fprintf(out,"%f %d %d %d\n", (cfg->steps.x),(cfg->dim.x),(cfg->crop0.x),(cfg->crop1.x));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.y),(cfg->dim.y),(cfg->crop0.y),(cfg->crop1.y));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.z),(cfg->dim.z),(cfg->crop0.z),(cfg->crop1.z));
     fprintf(out,"%d\n", (cfg->medianum));
     for(i=0;i<cfg->medianum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->prop[i].mus),(cfg->prop[i].g),(cfg->prop[i].mua),(cfg->prop[i].n));
     }
     fprintf(out,"%d", (cfg->detnum));
     for(i=0;i<cfg->detnum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->detpos[i].x),(cfg->detpos[i].y),(cfg->detpos[i].z),(cfg->detpos[i].w));
     }
}

/**
 * @brief Load media index data volume (.bin or .vol) to the memory
 *
 * @param[in] filename: file name to the binary volume data (support 1,2 and 4 bytes per voxel)
 * @param[in] cfg: simulation configuration
 */

void mcx_loadvolume(char *filename,Config *cfg){
     unsigned int i,datalen,res;
     unsigned char *inputvol=NULL;
     FILE *fp;
     
     if(strstr(filename,".json")!=NULL){
         int status;
         Grid3D grid={&(cfg->vol),&(cfg->dim),{1.f,1.f,1.f},cfg->isrowmajor};
	 if(cfg->issrcfrom0) memset(&(grid.orig.x),0,sizeof(float3));
         status=mcx_load_jsonshapes(&grid,filename);
	 if(status){
	     MCX_ERROR(status,mcx_last_shapeerror());
	 }
	 return;
     }
     fp=fopen(filename,"rb");
     if(fp==NULL){
     	     mcx_error(-5,"the specified binary volume file does not exist",__FILE__,__LINE__);
     }
     if(cfg->vol){
     	     free(cfg->vol);
     	     cfg->vol=NULL;
     }
     datalen=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     cfg->vol=(unsigned int*)malloc(sizeof(unsigned int)*datalen);
     if(cfg->mediabyte==MEDIA_AS_F2H)
         inputvol=(unsigned char*)malloc(sizeof(unsigned char)*(datalen<<3));
     else if(cfg->mediabyte>=4)
         inputvol=(unsigned char*)(cfg->vol);
     else
         inputvol=(unsigned char*)malloc(sizeof(unsigned char)*cfg->mediabyte*datalen);
     res=fread(inputvol,sizeof(unsigned char)*(cfg->mediabyte==MEDIA_AS_F2H? 8 : MIN(cfg->mediabyte,4)),datalen,fp);
     fclose(fp);
     if(res!=datalen){
     	 mcx_error(-6,"file size does not match specified dimensions",__FILE__,__LINE__);
     }
     if(cfg->mediabyte==1){  /*convert all format into 4-byte int index*/
       unsigned char *val=inputvol;
       for(i=0;i<datalen;i++)
         cfg->vol[i]=val[i];
     }else if(cfg->mediabyte==2){
       unsigned short *val=(unsigned short *)inputvol;
       for(i=0;i<datalen;i++)
         cfg->vol[i]=val[i];
     }else if(cfg->mediabyte==MEDIA_MUA_FLOAT){
       union{
           float f;
	   uint  i;
       } f2i;
       float *val=(float *)inputvol;
       for(i=0;i<datalen;i++){
         f2i.f=val[i]*cfg->unitinmm;
         cfg->vol[i]=f2i.i;
       }
     }else if(cfg->mediabyte==MEDIA_AS_F2H){
        float *val=(float *)inputvol;
	union{
	    float f[2];
	    unsigned int i[2];
	    unsigned short h[2];
	} f2h;
	unsigned short tmp;
        for(i=0;i<datalen;i++){
	    f2h.f[0]=val[i<<1]*cfg->unitinmm;
	    f2h.f[1]=val[(i<<1)+1]*cfg->unitinmm;

	    f2h.h[0] = (f2h.i[0] >> 31) << 5;
	    tmp = (f2h.i[0] >> 23) & 0xff;
	    tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
	    f2h.h[0] = (f2h.h[0] | tmp) << 10;
	    f2h.h[0] |= (f2h.i[0] >> 13) & 0x3ff;

	    f2h.h[1] = (f2h.i[1] >> 31) << 5;
	    tmp = (f2h.i[1] >> 23) & 0xff;
	    tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
	    f2h.h[1] = (f2h.h[1] | tmp) << 10;
	    f2h.h[1] |= (f2h.i[1] >> 13) & 0x3ff;

            cfg->vol[i]=f2h.i[0];
	}
     }
     if(cfg->mediabyte<=4)
       for(i=0;i<datalen;i++){
         if(cfg->vol[i]>=cfg->medianum)
            mcx_error(-6,"medium index exceeds the specified medium types",__FILE__,__LINE__);
     }
     if(cfg->mediabyte<4 || cfg->mediabyte==MEDIA_AS_F2H)
         free(inputvol);
}


/**
 * @brief Load previously saved photon seeds from an .mch file for replay
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_loadseedfile(Config *cfg){
    History his;
    FILE *fp=fopen(cfg->seedfile,"rb");
    if(fp==NULL)
        mcx_error(-7,"can not open the specified history file",__FILE__,__LINE__);
    if(fread(&his,sizeof(History),1,fp)!=1)
        mcx_error(-7,"error when reading the history file",__FILE__,__LINE__);
    if(his.savedphoton==0 || his.seedbyte==0){
	mcx_error(-7,"history file does not contain seed data, please re-run your simulation with '-q 1'",__FILE__,__LINE__);
    }
    if(his.maxmedia!=cfg->medianum-1)
        mcx_error(-7,"the history file was generated with a different media setting",__FILE__,__LINE__);
    if(fseek(fp,his.savedphoton*his.colcount*sizeof(float),SEEK_CUR))
        mcx_error(-7,"illegal history file",__FILE__,__LINE__);
    cfg->replay.seed=malloc(his.savedphoton*his.seedbyte);
    if(cfg->replay.seed==NULL)
        mcx_error(-7,"can not allocate memory",__FILE__,__LINE__);
    if(fread(cfg->replay.seed,his.seedbyte,his.savedphoton,fp)!=his.savedphoton)
        mcx_error(-7,"error when reading the seed data",__FILE__,__LINE__);
    cfg->seed=SEED_FROM_FILE;
    cfg->nphoton=his.savedphoton;

    if(cfg->outputtype==otJacobian || cfg->outputtype==otWP || cfg->outputtype==otDCS ){ //cfg->replaydet>0
       int i,j, hasdetid=0, offset;
       float plen, *ppath;
       hasdetid=SAVE_DETID(his.savedetflag);
       offset=SAVE_NSCAT(his.savedetflag)*his.maxmedia;

       if(((!hasdetid) && cfg->detnum>1) || !SAVE_PPATH(his.savedetflag))
           mcx_error(-7,"please rerun the baseline simulation and save detector ID (D) and partial-path (P) using '-w DP'",__FILE__,__LINE__);

       ppath=(float*)malloc(his.savedphoton*his.colcount*sizeof(float));
       cfg->replay.weight=(float*)malloc(his.savedphoton*sizeof(float));
       cfg->replay.tof=(float*)calloc(his.savedphoton,sizeof(float));
       cfg->replay.detid=(int*)calloc(his.savedphoton,sizeof(int));
       fseek(fp,sizeof(his),SEEK_SET);
       if(fread(ppath,his.colcount*sizeof(float),his.savedphoton,fp)!=his.savedphoton)
           mcx_error(-7,"error when reading the seed data",__FILE__,__LINE__);

       cfg->nphoton=0;
       for(i=0;i<his.savedphoton;i++)
           if(cfg->replaydet<=0 || cfg->replaydet==(int)(ppath[i*his.colcount])){
               if(i!=cfg->nphoton)
                   memcpy((char *)(cfg->replay.seed)+cfg->nphoton*his.seedbyte, (char *)(cfg->replay.seed)+i*his.seedbyte, his.seedbyte);
               cfg->replay.weight[cfg->nphoton]=1.f;
               cfg->replay.detid[cfg->nphoton]=(hasdetid) ? (int)(ppath[i*his.colcount]): 1;
               for(j=hasdetid;j<his.maxmedia+hasdetid;j++){
	           plen=ppath[i*his.colcount+offset+j]*his.unitinmm;
                   cfg->replay.weight[cfg->nphoton]*=expf(-cfg->prop[j-hasdetid+1].mua*plen);
                   cfg->replay.tof[cfg->nphoton]+=plen*R_C0*cfg->prop[j-hasdetid+1].n;
               }
               if(cfg->replay.tof[cfg->nphoton]<cfg->tstart || cfg->replay.tof[cfg->nphoton]>cfg->tend) /*need to consider -g*/
                   continue;
               cfg->nphoton++;
           }
	free(ppath);
        cfg->replay.seed=realloc(cfg->replay.seed, cfg->nphoton*his.seedbyte);
        cfg->replay.weight=(float*)realloc(cfg->replay.weight, cfg->nphoton*sizeof(float));
        cfg->replay.tof=(float*)realloc(cfg->replay.tof, cfg->nphoton*sizeof(float));
        cfg->replay.detid=(int*)realloc(cfg->replay.detid, cfg->nphoton*sizeof(int));
	cfg->minenergy=0.f;
    }
    fclose(fp);
}

/**
 * @brief Convert a row-major (C/C++) array to a column-major (MATLAB/FORTRAN) array
 *
 * @param[in,out] vol: a 3D array (wrapped in 1D) to be converted
 * @param[in] dim: the dimensions of the 3D array
 */

void  mcx_convertrow2col(unsigned int **vol, uint3 *dim){
     uint x,y,z;
     unsigned int dimxy,dimyz;
     unsigned int *newvol=NULL;
     
     if(*vol==NULL || dim->x==0 || dim->y==0 || dim->z==0){
     	return;
     }     
     newvol=(unsigned int*)malloc(sizeof(unsigned int)*dim->x*dim->y*dim->z);
     dimxy=dim->x*dim->y;
     dimyz=dim->y*dim->z;
     for(x=0;x<dim->x;x++)
      for(y=0;y<dim->y;y++)
       for(z=0;z<dim->z;z++){
       		newvol[z*dimxy+y*dim->x+x]=*vol[x*dimyz+y*dim->z+z];
       }
     free(*vol);
     *vol=newvol;
}

/**
 * @brief Pre-label the voxel near a detector for easy photon detection
 *
 * This function preprocess the volume and detector data and add the detector ID to the
 * upper 16bits of the voxel that the detector encompasses. If two detectors cover the same
 * voxel, the later one will overwrite the ID of the 1st one. In MCX kernel, the detector
 * coverage is tested for all detectors despite the label written (only used as a binary mask)
 *
 * @param[in] cfg: simulation configuration
 */

void  mcx_maskdet(Config *cfg){
     uint d,dx,dy,dz,idx1d,zi,yi,c,count;
     float x,y,z,ix,iy,iz,rx,ry,rz,d2,mind2,d2max;
     unsigned int *padvol;
     const float corners[8][3]={{0.f,0.f,0.f},{1.f,0.f,0.f},{0.f,1.f,0.f},{0.f,0.f,1.f},
                                {1.f,1.f,0.f},{1.f,0.f,1.f},{0.f,1.f,1.f},{1.f,1.f,1.f}};
     
     dx=cfg->dim.x+2;
     dy=cfg->dim.y+2;
     dz=cfg->dim.z+2;
     
     /*handling boundaries in a volume search is tedious, I first pad vol by a layer of zeros,
       then I don't need to worry about boundaries any more*/

     padvol=(unsigned int*)calloc(dx*dy*sizeof(unsigned int),dz);

     for(zi=1;zi<=cfg->dim.z;zi++)
        for(yi=1;yi<=cfg->dim.y;yi++)
	        memcpy(padvol+zi*dy*dx+yi*dx+1,cfg->vol+(zi-1)*cfg->dim.y*cfg->dim.x+(yi-1)*cfg->dim.x,cfg->dim.x*sizeof(int));

     /**
        The goal here is to find a set of voxels for each 
	detector so that the intersection between a sphere
	of R=cfg->detradius,c0=cfg->detpos[d] and the object 
	surface (or bounding box) is fully covered.
     */
     for(d=0;d<cfg->detnum;d++){                             /*loop over each detector*/
        count=0;
        d2max=(cfg->detpos[d].w+1.7321f)*(cfg->detpos[d].w+1.7321f);
        for(z=-cfg->detpos[d].w-1.f;z<=cfg->detpos[d].w+1.f;z+=0.5f){   /*search in a cube with edge length 2*R+3*/
           iz=z+cfg->detpos[d].z;
           for(y=-cfg->detpos[d].w-1.f;y<=cfg->detpos[d].w+1.f;y+=0.5f){
              iy=y+cfg->detpos[d].y;
              for(x=-cfg->detpos[d].w-1.f;x<=cfg->detpos[d].w+1.f;x+=0.5f){
	         ix=x+cfg->detpos[d].x;

		 if(iz<0||ix<0||iy<0||ix>=cfg->dim.x||iy>=cfg->dim.y||iz>=cfg->dim.z||
		    x*x+y*y+z*z > (cfg->detpos[d].w+1.f)*(cfg->detpos[d].w+1.f))
		     continue;
		 mind2=VERY_BIG;
                 for(c=0;c<8;c++){ /*test each corner of a voxel*/
			rx=(int)ix-cfg->detpos[d].x+corners[c][0];
			ry=(int)iy-cfg->detpos[d].y+corners[c][1];
			rz=(int)iz-cfg->detpos[d].z+corners[c][2];
			d2=rx*rx+ry*ry+rz*rz;
		 	if(d2>d2max){ /*R+sqrt(3) to make sure the circle is fully corvered*/
				mind2=VERY_BIG;
		     		break;
			}
			if(d2<mind2) mind2=d2;
		 }
		 if(mind2==VERY_BIG || mind2>=cfg->detpos[d].w*cfg->detpos[d].w) continue;
		 idx1d=((int)(iz+1.f)*dy*dx+(int)(iy+1.f)*dx+(int)(ix+1.f)); /*1.f comes from the padded layer*/

		 if(padvol[idx1d])  /*looking for a voxel on the interface or bounding box*/
                  if(!(padvol[idx1d+1]&&padvol[idx1d-1]&&padvol[idx1d+dx]&&padvol[idx1d-dx]&&padvol[idx1d+dy*dx]&&padvol[idx1d-dy*dx]&&
		     padvol[idx1d+dx+1]&&padvol[idx1d+dx-1]&&padvol[idx1d-dx+1]&&padvol[idx1d-dx-1]&&
		     padvol[idx1d+dy*dx+1]&&padvol[idx1d+dy*dx-1]&&padvol[idx1d-dy*dx+1]&&padvol[idx1d-dy*dx-1]&&
		     padvol[idx1d+dy*dx+dx]&&padvol[idx1d+dy*dx-dx]&&padvol[idx1d-dy*dx+dx]&&padvol[idx1d-dy*dx-dx]&&
		     padvol[idx1d+dy*dx+dx+1]&&padvol[idx1d+dy*dx+dx-1]&&padvol[idx1d+dy*dx-dx+1]&&padvol[idx1d+dy*dx-dx-1]&&
		     padvol[idx1d-dy*dx+dx+1]&&padvol[idx1d-dy*dx+dx-1]&&padvol[idx1d-dy*dx-dx+1]&&padvol[idx1d-dy*dx-dx-1])){
		          cfg->vol[((int)iz*cfg->dim.y*cfg->dim.x+(int)iy*cfg->dim.x+(int)ix)] |= DET_MASK;/*set the highest bit to 1*/
                          count++;
	          }
	       }
	   }
        }
        if(cfg->issavedet && count==0)
              MCX_FPRINTF(stderr,S_RED"WARNING: detector %d is not located on an interface, please check coordinates.\n"S_RESET,d+1);
     }

     free(padvol);
}

/**
 * @brief Save the pre-masked volume (with detector ID) to an nii file
 *
 * To test the results, you should use -M to dump the det-mask, load 
 * it in matlab, and plot the interface containing the detector with
 * pcolor() (has the matching index), and then draw a circle with the
 * radius and center set in the input file. the pixels should completely
 * cover the circle.
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_dumpmask(Config *cfg){
     char fname[MAX_PATH_LENGTH];
     if(cfg->rootpath[0])
         sprintf(fname,"%s%c%s_vol",cfg->rootpath,pathsep,cfg->session);
     else
         sprintf(fname,"%s_vol",cfg->session);

     mcx_savenii((float *)cfg->vol, cfg->dim.x*cfg->dim.y*cfg->dim.z, fname, NIFTI_TYPE_UINT32, ofNifti, cfg);
     if(cfg->isdumpmask==1){ /*if dumpmask>1, simulation will also run*/
         MCX_FPRINTF(cfg->flog,"volume mask is saved as uint16 format in %s",fname);
         exit(0);
     }
}

/**
 * @brief Print a progress bar
 *
 * When -D P is specified, this function prints and update a progress bar.
 *
 * @param[in] percent: the percentage value from 1 to 100
 * @param[in] cfg: simulation configuration
 */

void mcx_progressbar(float percent, Config *cfg){
    unsigned int percentage, j,colwidth=79;
    static unsigned int oldmarker=0xFFFFFFFF;

#ifndef MCX_CONTAINER
  #ifdef TIOCGWINSZ
    struct winsize ttys={0,0,0,0};
    ioctl(0, TIOCGWINSZ, &ttys);
    colwidth=ttys.ws_col;
    if(colwidth==0)
         colwidth=79;
  #endif
#endif
    percent=MIN(percent,1.f);

    percentage=percent*(colwidth-18);

    if(percentage != oldmarker){
        if(percent!=-0.f)
	    for(j=0;j<colwidth;j++)     MCX_FPRINTF(stdout,"\b");
        oldmarker=percentage;
        MCX_FPRINTF(stdout,S_YELLOW"Progress: [");
        for(j=0;j<percentage;j++)      MCX_FPRINTF(stdout,"=");
        MCX_FPRINTF(stdout,(percentage<colwidth-18) ? ">" : "=");
        for(j=percentage;j<colwidth-18;j++) MCX_FPRINTF(stdout," ");
        MCX_FPRINTF(stdout,"] %3d%%"S_RESET,(int)(percent*100));
#ifdef MCX_CONTAINER
        mcx_matlab_flush();
#else
        fflush(stdout);
#endif
    }
}

/**
 * @brief Function to read a single parameter value followed by a command line option
 *
 * This function reads different types of parameter values following a command line option.
 *
 * @param[in] argc: the number of total command line parameters
 * @param[in] argv: the pointer to all command line options
 * @param[in] id: which parameter to be parsed
 * @param[out] output: the pointer to which the parsed value to be written
 * @param[in] type: the type of data support char, int, float, string, bytenumlist, floatlist
 */

int mcx_readarg(int argc, char *argv[], int id, void *output,const char *type){
     /*
         when a binary option is given without a following number (0~1), 
         we assume it is 1
     */
     if(strcmp(type,"char")==0 && (id>=argc-1||(argv[id+1][0]<'0'||argv[id+1][0]>'9'))){
	*((char*)output)=1;
	return id;
     }
     if(id<argc-1){
         if(strcmp(type,"char")==0)
             *((char*)output)=atoi(argv[id+1]);
	 else if(strcmp(type,"int")==0)
             *((int*)output)=atoi(argv[id+1]);
	 else if(strcmp(type,"float")==0)
             *((float*)output)=atof(argv[id+1]);
	 else if(strcmp(type,"string")==0)
	     strcpy((char *)output,argv[id+1]);
         else if(strcmp(type,"bytenumlist")==0){
             char *nexttok,*numlist=(char *)output;
             int len=0,i;
             nexttok=strtok(argv[id+1]," ,;");
             while(nexttok){
                 numlist[len++]=(char)(atoi(nexttok)); /*device id<256*/
                 for(i=0;i<len-1;i++) /* remove duplicaetd ids */
                    if(numlist[i]==numlist[len-1]){
                       numlist[--len]='\0';
                       break;
                    }
                 nexttok=strtok(NULL," ,;");
                 /*if(len>=MAX_DEVICE) break;*/
             }
         }else if(strcmp(type,"floatlist")==0){
             char *nexttok;
             float *numlist=(float *)output;
             int len=0;   
             nexttok=strtok(argv[id+1]," ,;");
             while(nexttok){
                 numlist[len++]=atof(nexttok); /*device id<256*/
                 nexttok=strtok(NULL," ,;");
             }
	 }
     }else{
     	 mcx_error(-1,"incomplete input",__FILE__,__LINE__);
     }
     return id+1;
}

/**
 * @brief Test if a long command line option is supported
 *
 * This function returns 1 if a long option is found, and 0 otherwise
 *
 * @param[in] opt: the long command line option string
 */

int mcx_remap(char *opt){
    int i=0;
    while(shortopt[i]!='\0'){
	if(strcmp(opt,fullopt[i])==0){
		opt[1]=shortopt[i];
		if(shortopt[i]!='-')
                    opt[2]='\0';
		return 0;
	}
	i++;
    }
    return 1;
}

/**
 * @brief Main function to read user command line options
 *
 * This function process user command line inputs and parse all short and long options.
 *
 * @param[in] argc: the number of total command line parameters
 * @param[in] argv: the pointer to all command line options
 * @param[in] cfg: simulation configuration
 */

void mcx_parsecmd(int argc, char* argv[], Config *cfg){
     int i=1,isinteractive=1,issavelog=0;
     char filename[MAX_PATH_LENGTH]={0}, *jsoninput=NULL;
     char logfile[MAX_PATH_LENGTH]={0};
     float np=0.f;

     if(argc<=1){
     	mcx_usage(cfg,argv[0]);
     	exit(0);
     }
     while(i<argc){
     	    if(argv[i][0]=='-'){
		if(argv[i][1]=='-'){
			if(mcx_remap(argv[i])){
				mcx_error(-2,"unknown verbose option",__FILE__,__LINE__);
			}
		}
		if(argv[i][1]<='z' && argv[i][1]>='A')
		     flagset[(int)(argv[i][1])]=1;
	        switch(argv[i][1]){
		     case 'h': 
		                mcx_usage(cfg,argv[0]);
				exit(0);
		     case 'i':
				if(filename[0]){
					mcx_error(-2,"you can not specify both interactive mode and config file",__FILE__,__LINE__);
				}
		     		isinteractive=1;
				break;
		     case 'f': 
		     		isinteractive=0;
				if(i<argc-1 && argv[i+1][0]=='{'){
					jsoninput=argv[i+1];
					i++;
				}else
		     	        	i=mcx_readarg(argc,argv,i,filename,"string");
				break;
		     case 'n':
		     	        i=mcx_readarg(argc,argv,i,&(np),"float");
				cfg->nphoton=(size_t)np;
		     	        break;
		     case 't':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->nthread),"int");
		     	        break;
                     case 'T':
                               	i=mcx_readarg(argc,argv,i,&(cfg->nblocksize),"int");
                               	break;
		     case 's':
		     	        i=mcx_readarg(argc,argv,i,cfg->session,"string");
		     	        break;
		     case 'a':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isrowmajor),"char");
		     	        break;
		     case 'q':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->issaveseed),"char");
		     	        break;
		     case 'g':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->maxgate),"int");
		     	        break;
		     case 'b':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isreflect),"char");
				cfg->isref3=cfg->isreflect;
		     	        break;
                     case 'B':
                                if(i<argc+1)
				    strncpy(cfg->bc,argv[i+1],8);
				i++;
                               	break;
		     case 'd':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->issavedet),"char");
		     	        break;
		     case 'm':
		                i=mcx_readarg(argc,argv,i,&(cfg->ismomentum),"char");
				if (cfg->ismomentum) cfg->issavedet=1;
				break;
		     case 'r':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->respin),"int");
		     	        break;
		     case 'S':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->issave2pt),"char");
		     	        break;
		     case 'p':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->printnum),"int");
		     	        break;
                     case 'e':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->minenergy),"float");
                                break;
		     case 'U':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isnormalized),"char");
		     	        break;
                     case 'R':
                                i=mcx_readarg(argc,argv,i,&(cfg->sradius),"float");
                                break;
                     case 'u':
                                i=mcx_readarg(argc,argv,i,&(cfg->unitinmm),"float");
                                break;
                     case 'l':
                                issavelog=1;
                                break;
		     case 'L':
                                cfg->isgpuinfo=2;
		                break;
		     case 'I':
                                cfg->isgpuinfo=1;
		                break;
                     case 'G':
                                if(mcx_isbinstr(argv[i+1])){
                                    i=mcx_readarg(argc,argv,i,cfg->deviceid,"string");
                                    break;
                                }else{
                                    i=mcx_readarg(argc,argv,i,&(cfg->gpuid),"int");
                                    memset(cfg->deviceid,'0',MAX_DEVICE);
                                    if(cfg->gpuid>0 && cfg->gpuid<MAX_DEVICE)
                                         cfg->deviceid[cfg->gpuid-1]='1';
                                    else
                                         mcx_error(-2,"GPU id can not be more than 256",__FILE__,__LINE__);
                                    break;
                                }
                     case 'W':
                                i=mcx_readarg(argc,argv,i,cfg->workload,"floatlist");
                                break;
                     case 'z':
                                i=mcx_readarg(argc,argv,i,&(cfg->issrcfrom0),"char");
                                break;
		     case 'M':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isdumpmask),"char");
		     	        break;
                     case 'Y':
                                i=mcx_readarg(argc,argv,i,&(cfg->replaydet),"int");
                                break;
		     case 'H':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->maxdetphoton),"int");
		     	        break;
                     case 'P':
                                cfg->shapedata=argv[++i];
                                break;
                     case 'A':
                                i=mcx_readarg(argc,argv,i,&(cfg->autopilot),"char");
                                break;
                     case 'E':
				if(i<argc-1 && strstr(argv[i+1],".mch")!=NULL){ /*give an mch file to initialize the seed*/
#if defined(USE_LL5_RAND)
					mcx_error(-1,"seeding file is not supported in this binary",__FILE__,__LINE__);
#else
                                        i=mcx_readarg(argc,argv,i,cfg->seedfile,"string");
					cfg->seed=SEED_FROM_FILE;
#endif
		     	        }else
					i=mcx_readarg(argc,argv,i,&(cfg->seed),"int");
		     	        break;
                     case 'O':
                                i=mcx_readarg(argc,argv,i,&(cfg->outputtype),"string");
				if(mcx_lookupindex(&(cfg->outputtype), outputtype)){
                                        mcx_error(-2,"the specified output data type is not recognized",__FILE__,__LINE__);
                                }
                                break;
                     case 'k':
                                i=mcx_readarg(argc,argv,i,&(cfg->voidtime),"int");
                                break;
		     case 'V':
		     	        i=mcx_readarg(argc,argv,i,&(cfg->isspecular),"char");
		     	        break;
                     case 'v':
                                mcx_version(cfg);
				break;
                     case 'D':
                                if(i+1<argc && isalpha(argv[i+1][0]) )
                                        cfg->debuglevel=mcx_parsedebugopt(argv[++i],debugflag);
                                else
                                        i=mcx_readarg(argc,argv,i,&(cfg->debuglevel),"int");
                                break;
		     case 'F':
                                if(i>=argc)
                                        mcx_error(-1,"incomplete input",__FILE__,__LINE__);
                                if((cfg->outputformat=mcx_keylookup(argv[++i], outputformat))<0)
                                        mcx_error(-2,"the specified output data type is not recognized",__FILE__,__LINE__);
                                break;
		     case 'x':
 		                i=mcx_readarg(argc,argv,i,&(cfg->issaveexit),"char");
 				if (cfg->issaveexit) cfg->issavedet=1;
 				break;
		     case 'X':
 		                i=mcx_readarg(argc,argv,i,&(cfg->issaveref),"char");
 				if (cfg->issaveref) cfg->issaveref=1;
 				break;
		     case 'w':
			        if(i+1<argc && isalpha(argv[i+1][0]) ){
				    cfg->savedetflag=mcx_parsedebugopt(argv[++i],saveflag);
			        }else
				    i=mcx_readarg(argc,argv,i,&(cfg->savedetflag),"int");
 				break;
		     case '-':  /*additional verbose parameters*/
                                if(strcmp(argv[i]+2,"maxvoidstep")==0)
                                     i=mcx_readarg(argc,argv,i,&(cfg->maxvoidstep),"int");
                                else if(strcmp(argv[i]+2,"maxjumpdebug")==0)
                                     i=mcx_readarg(argc,argv,i,&(cfg->maxjumpdebug),"int");
                                else if(strcmp(argv[i]+2,"gscatter")==0)
                                     i=mcx_readarg(argc,argv,i,&(cfg->gscatter),"int");
                                else if(strcmp(argv[i]+2,"mediabyte")==0){
				     if(i+1<argc && isalpha(argv[i+1][0]) ){
				         cfg->mediabyte=mcx_keylookup(argv[++i],mediaformat);
					 if(cfg->mediabyte==-1)
					     MCX_ERROR(-1,"Unsupported media format.");
					 cfg->mediabyte=mediaformatid[cfg->mediabyte];
			             }else
				         i=mcx_readarg(argc,argv,i,&(cfg->mediabyte),"int");
                                }else if(strcmp(argv[i]+2,"faststep")==0)
                                     i=mcx_readarg(argc,argv,i,&(cfg->faststep),"char");
                                else if(strcmp(argv[i]+2,"root")==0)
                                     i=mcx_readarg(argc,argv,i,cfg->rootpath,"string");
                                else if(strcmp(argv[i]+2,"reflectin")==0)
                                     i=mcx_readarg(argc,argv,i,&(cfg->isrefint),"char");
                                else if(strcmp(argv[i]+2,"internalsrc")==0)
		                     i=mcx_readarg(argc,argv,i,&(cfg->internalsrc),"int");
                                else
                                     MCX_FPRINTF(cfg->flog,"unknown verbose option: --%s\n",argv[i]+2);
		     	        break;
		}
	    }
	    i++;
     }
     if(issavelog && cfg->session[0]){
          sprintf(logfile,"%s.log",cfg->session);
          cfg->flog=fopen(logfile,"wt");
          if(cfg->flog==NULL){
		cfg->flog=stdout;
		MCX_FPRINTF(cfg->flog,"unable to save to log file, will print from stdout\n");
          }
     }
     if((cfg->outputtype==otJacobian ||cfg->outputtype==otWP || cfg->outputtype==otDCS) && cfg->seed!=SEED_FROM_FILE)
         MCX_ERROR(-1,"Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.");

     if(cfg->isgpuinfo!=2){ /*print gpu info only*/
	  if(isinteractive){
             mcx_readconfig((char*)"",cfg);
	  }else if(jsoninput){
     	     mcx_readconfig(jsoninput,cfg);
	  }else{
             mcx_readconfig(filename,cfg);
          }
     }
}

/**
 * @brief Parse the debug flag in the letter format
 *
 * The debug flag following the -D can be either a string format, or numerical format.
 * This function converts the string debug flags into number format
 *
 * @param[in] debugopt: string following the -D parameter
 * @param[out] debugflag: the numerical format of the debug flag
 */

int mcx_parsedebugopt(char *debugopt,const char *debugflag){
    char *c=debugopt,*p;
    int debuglevel=0;

    while(*c){
       p=strchr(debugflag, ((*c<='z' && *c>='a') ? *c-'a'+'A' : *c) );
       if(p!=NULL)
          debuglevel |= (1 << (p-debugflag));
       c++;
    }
    return debuglevel;
}

/**
 * @brief Look up a string in a string list and return the index
 *
 * @param[in] origkey: string to be looked up
 * @param[out] table: the dictionary where the string is searched
 * @return if found, return the index of the string in the dictionary, otherwise -1.
 */

int mcx_keylookup(char *origkey, const char *table[]){
    int i=0;
    char *key=malloc(strlen(origkey)+1);
    memcpy(key,origkey,strlen(origkey)+1);
    while(key[i]){
        key[i]=tolower(key[i]);
	i++;
    }
    i=0;
    while(table[i] && table[i][0]!='\0'){
	if(strcmp(key,table[i])==0){
	        free(key);
		return i;
	}
	i++;
    }
    free(key);
    return -1;
}

/**
 * @brief Look up a single character in a string
 *
 * @param[in] key: character to be looked up
 * @param[out] index: the dictionary string where the char is searched
 * @return if found, return 0; otherwise, return 1
 */

int mcx_lookupindex(char *key, const char *index){
    int i=0;
    while(index[i]!='\0'){
        if(tolower(*key)==index[i]){
                *key=i;
                return 0;
        }
        i++;
    }
    return 1;
}

/**
 * @brief Print MCX software version
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_version(Config *cfg){
    const char ver[]="$Rev::      $2019.4";
    int v=0;
    sscanf(ver,"$Rev::%x",&v);
    MCX_FPRINTF(cfg->flog, "MCX Revision %x\n",v);
    exit(0);
}

/**
 * @brief Test if a string contains only '0' and '1'
 *
 * @param[in] str: string to be tested
 */

int mcx_isbinstr(const char * str){
    int i, len=strlen(str);
    if(len==0)
        return 0;
    for(i=0;i<len;i++)
        if(str[i]!='0' && str[i]!='1')
	   return 0;
    return 1;
}

/**
 * @brief Run MCX simulations from a JSON input in a persistent session
 *
 * @param[in] jsonstr: a string in the JSON format, the content of the .json input file
 */

int mcx_run_from_json(char *jsonstr){
     Config  mcxconfig;            /** mcxconfig: structure to store all simulation parameters */
     GPUInfo *gpuinfo=NULL;        /** gpuinfo: structure to store GPU information */
     unsigned int activedev=0;     /** activedev: count of total active GPUs to be used */

     mcx_initcfg(&mcxconfig);
     mcx_readconfig(jsonstr, &mcxconfig);

     if(!(activedev=mcx_list_gpu(&mcxconfig,&gpuinfo))){
         mcx_error(-1,"No GPU device found\n",__FILE__,__LINE__);
     }

#ifdef _OPENMP
     omp_set_num_threads(activedev);
     #pragma omp parallel
     {
#endif
         mcx_run_simulation(&mcxconfig,gpuinfo); 
#ifdef _OPENMP
     }
#endif

     mcx_cleargpuinfo(&gpuinfo);
     mcx_clearcfg(&mcxconfig);
     return 0;
}

/**
 * @brief Print MCX output header
 *
 * @param[in] cfg: simulation configuration
 */

void mcx_printheader(Config *cfg){
    MCX_FPRINTF(cfg->flog,S_GREEN"\
###############################################################################\n\
#                      Monte Carlo eXtreme (MCX) -- CUDA                      #\n\
#          Copyright (c) 2009-2019 Qianqian Fang <q.fang at neu.edu>          #\n\
#                             http://mcx.space/                               #\n\
#                                                                             #\n\
# Computational Optics & Translational Imaging (COTI) Lab- http://fanglab.org #\n\
#            Department of Bioengineering, Northeastern University            #\n\
###############################################################################\n\
#    The MCX Project is funded by the NIH/NIGMS under grant R01-GM114365      #\n\
###############################################################################\n\
$Rev::      $2019.4 $Date::                       $ by $Author::              $\n\
###############################################################################\n"S_RESET);
}

/**
 * @brief Print MCX help information
 *
 * @param[in] cfg: simulation configuration structure
 * @param[in] exename: path and name of the mcx executable
 */

void mcx_usage(Config *cfg,char *exename){
     mcx_printheader(cfg);
     printf("\n\
usage: %s <param1> <param2> ...\n\
where possible parameters include (the first value in [*|*] is the default)\n\
\n"S_BOLD S_CYAN"\
== Required option ==\n"S_RESET"\
 -f config     (--input)       read an input file in .json or .inp format\n\
\n"S_BOLD S_CYAN"\
== MC options ==\n"S_RESET"\
 -n [0|int]    (--photon)      total photon number (exponential form accepted)\n\
                               max accepted value:9.2234e+18 on 64bit systems\n\
 -r [1|+/-int] (--repeat)      if positive, repeat by r times,total= #photon*r\n\
                               if negative, divide #photon into r subsets\n\
 -b [1|0]      (--reflect)     1 to reflect photons at ext. boundary;0 to exit\n\
 -B '______'   (--bc)          per-face boundary condition (BC), 6 letters for\n\
    /case insensitive/         bounding box faces at -x,-y,-z,+x,+y,+z axes;\n\
			       overwrite -b if given. \n\
			       each letter can be one of the following:\n\
			       '_': undefined, fallback to -b\n\
			       'r': like -b 1, Fresnel reflection BC\n\
			       'a': like -b 0, total absorption BC\n\
			       'm': mirror or total reflection BC\n\
			       'c': cyclic BC, enter from opposite face\n\
 -u [1.|float] (--unitinmm)    defines the length unit for the grid edge\n\
 -U [1|0]      (--normalize)   1 to normalize flux to unitary; 0 save raw\n\
 -E [0|int|mch](--seed)        set random-number-generator seed, -1 to generate\n\
                               if an mch file is followed, MCX \"replays\" \n\
                               the detected photon; the replay mode can be used\n\
                               to calculate the mua/mus Jacobian matrices\n\
 -z [0|1]      (--srcfrom0)    1 volume origin is [0 0 0]; 0: origin at [1 1 1]\n\
 -R [-2|float] (--skipradius)  -2: use atomics for the entire domain (default)\n\
                                0: vanilla MCX, no atomic operations\n\
                               >0: radius in which use shared-memory atomics\n\
                               -1: use crop0/crop1 to determine atomic zone\n\
 -k [1|0]      (--voidtime)    when src is outside, 1 enables timer inside void\n\
 -Y [0|int]    (--replaydet)   replay only the detected photons from a given \n\
                               detector (det ID starts from 1), used with -E \n\
			       if 0, replay all detectors and sum all Jacobians\n\
			       if -1, replay all detectors and save separately\n\
 -V [0|1]      (--specular)    1 source located in the background,0 inside mesh\n\
 -e [0.|float] (--minenergy)   minimum energy level to terminate a photon\n\
 -g [1|int]    (--gategroup)   number of time gates per run\n\
\n"S_BOLD S_CYAN"\
== GPU options ==\n"S_RESET"\
 -L            (--listgpu)     print GPU information only\n\
 -t [16384|int](--thread)      total thread number\n\
 -T [64|int]   (--blocksize)   thread number per block\n\
 -A [1|int]    (--autopilot)   1 let mcx decide thread/block size, 0 use -T/-t\n\
 -G [0|int]    (--gpu)         specify which GPU to use, list GPU by -L; 0 auto\n\
      or\n\
 -G '1101'     (--gpu)         using multiple devices (1 enable, 0 disable)\n\
 -W '50,30,20' (--workload)    workload for active devices; normalized by sum\n\
 -I            (--printgpu)    print GPU information and run program\n\
\n"S_BOLD S_CYAN"\
== Input options ==\n"S_RESET"\
 -P '{...}'    (--shapes)      a JSON string for additional shapes in the grid\n\
 -K [1|int|str](--mediabyte)   volume data format, use either a number or a str\n\
                               1 or byte: 0-128 tissue labels\n\
			       2 or short: 0-65535 (max to 4000) tissue labels\n\
			       4 or integer: integer tissue labels \n\
                             100 or muamus_float: 2x 32bit floats for mua/mus\n\
                             101 or mua_float: 1 float per voxel for mua\n\
			     102 or muamus_half: 2x 16bit float for mua/mus\n\
			     103 or asgn_byte: 4x byte gray-levels for mua/s/g/n\n\
			     104 or muamus_short: 2x short gray-levels for mua/s\n\
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array\n\
\n"S_BOLD S_CYAN"\
== Output options ==\n"S_RESET"\
 -s sessionid  (--session)     a string to label all output file names\n\
 -O [X|XFEJPM] (--outputtype)  X - output flux, F - fluence, E - energy deposit\n\
    /case insensitive/         J - Jacobian (replay mode),   P - scattering, \n\
			       event counts at each voxel (replay mode only)\n\
                               M - momentum transfer; \n\
 -d [1|0]      (--savedet)     1 to save photon info at detectors; 0 not save\n\
 -w [DP|DSPMXVW](--savedetflag)a string controlling detected photon data fields\n\
    /case insensitive/         1 D  output detector ID (1)\n\
                               2 S  output partial scat. even counts (#media)\n\
                               4 P  output partial path-lengths (#media)\n\
			       8 M  output momentum transfer (#media)\n\
			      16 X  output exit position (3)\n\
			      32 V  output exit direction (3)\n\
			      64 W  output initial weight (1)\n\
      combine multiple items by using a string, or add selected numbers together\n\
      by default, mcx only saves detector ID and partial-path data\n\
 -x [0|1]      (--saveexit)    1 to save photon exit positions and directions\n\
                               setting -x to 1 also implies setting '-d' to 1.\n\
			       same as adding 'XV' to -w.\n\
 -X [0|1]      (--saveref)     1 to save diffuse reflectance at the air-voxels\n\
                               right outside of the domain; if non-zero voxels\n\
			       appear at the boundary, pad 0s before using -X\n\
 -m [0|1]      (--momentum)    1 to save photon momentum transfer,0 not to save.\n\
                               same as adding 'M' to the -w flag\n\
 -q [0|1]      (--saveseed)    1 to save photon RNG seed for replay; 0 not save\n\
 -M [0|1]      (--dumpmask)    1 to dump detector volume masks; 0 do not save\n\
 -H [1000000] (--maxdetphoton) max number of detected photons\n\
 -S [1|0]      (--save2pt)     1 to save the flux field; 0 do not save\n\
 -F [mc2|...] (--outputformat) fluence data output format:\n\
                               mc2 - MCX mc2 format (binary 32bit float)\n\
                               nii - Nifti format\n\
                               hdr - Analyze 7.5 hdr/img format\n\
                               tx3 - GL texture data for rendering (GL_RGBA32F)\n\
\n"S_BOLD S_CYAN"\
== User IO options ==\n"S_RESET"\
 -h            (--help)        print this message\n\
 -v            (--version)     print MCX revision number\n\
 -l            (--log)         print messages to a log file instead\n\
 -i 	       (--interactive) interactive mode\n\
\n"S_BOLD S_CYAN"\
== Debug options ==\n"S_RESET"\
 -D [0|int]    (--debug)       print debug information (you can use an integer\n\
  or                           or a string by combining the following flags)\n\
 -D [''|RMP]                   1 R  debug RNG\n\
    /case insensitive/         2 M  store photon trajectory info\n\
                               4 P  print progress bar\n\
      combine multiple items by using a string, or add selected numbers together\n\
\n"S_BOLD S_CYAN"\
== Additional options ==\n"S_RESET"\
 --root         [''|string]    full path to the folder storing the input files\n\
 --gscatter     [1e9|int]      after a photon completes the specified number of\n\
                               scattering events, mcx then ignores anisotropy g\n\
                               and only performs isotropic scattering for speed\n\
 --internalsrc  [0|1]          set to 1 to skip entry search to speedup launch\n\
 --maxvoidstep  [1000|int]     maximum distance (in voxel unit) of a photon that\n\
                               can travel before entering the domain, if \n\
                               launched outside (i.e. a widefield source)\n\
 --maxjumpdebug [10000000|int] when trajectory is requested (i.e. -D M),\n\
                               use this parameter to set the maximum positions\n\
                               stored (default: 1e7)\n\
 --faststep [0|1]              1-use fast 1mm stepping, [0]-precise ray-tracing\n\
\n"S_BOLD S_CYAN"\
== Example ==\n"S_RESET"\
example: (autopilot mode)\n"S_GREEN"\
       %s -A 1 -n 1e7 -f input.inp -G 1 -D P\n"S_RESET"\
or (manual mode)\n"S_GREEN"\
       %s -t 16384 -T 64 -n 1e7 -f input.inp -s test -r 2 -g 10 -d 1 -w dpx -b 1 -G 1\n"S_RESET"\
or (use multiple devices - 1st,2nd and 4th GPUs - together with equal load)\n"S_GREEN"\
       %s -A -n 1e7 -f input.inp -G 1101 -W 10,10,10\n"S_RESET"\
or (use inline domain definition)\n"S_GREEN"\
       %s -f input.json -P '{\"Shapes\":[{\"ZLayers\":[[1,10,1],[11,30,2],[31,60,3]]}]}'"S_RESET"\n",
              exename,exename,exename,exename,exename);
}
