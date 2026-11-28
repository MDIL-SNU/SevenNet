/* Pair style e3gnn on the Apple GPU as a LAMMPS plugin, for `pip install lammps` without a rebuild. See build_plugin.sh.
   Two steps, as the pair style needs Python symbols at load time: this file loads libpython when the process has none
   (inside Python, a second libpython would be a second runtime), then the pair style itself, e3gnnmpscore.so.
   Part of SevenNet, MIT license (see LICENSE). Loaded by LAMMPS, the combined program is covered by the GPL
   of LAMMPS. */
#include <dlfcn.h>

#include <cstdio>
#include <string>

extern "C" {
/* The layout of lammpsplugin_t in lammpsplugin.h of LAMMPS, declared again here for binary compatibility, because
   `pip install lammps` does not ship that header. */
typedef void *(lammpsplugin_factory1)(void *);
typedef struct {
  const char *version;
  const char *style;
  const char *name;
  const char *info;
  const char *author;
  union {
    lammpsplugin_factory1 *v1;
    void *v2;
  } creator;
  void *handle;
} lammpsplugin_t;
typedef void (*lammpsplugin_regfunc)(lammpsplugin_t *, void *);

__attribute__((visibility("default"))) void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  if (!dlsym(RTLD_DEFAULT, "Py_IsInitialized") && !dlopen(SEVENN_LIBPYTHON, RTLD_NOW | RTLD_GLOBAL)) {
    fprintf(stderr, "e3gnn: cannot load %s: %s\n", SEVENN_LIBPYTHON, dlerror());
    return;
  }
  Dl_info self;
  dladdr(reinterpret_cast<void *>(&lammpsplugin_init), &self);
  std::string path(self.dli_fname);
  path = path.substr(0, path.rfind('/') + 1) + "e3gnnmpscore.so";
  void *core = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  void *create = core ? dlsym(core, "e3gnn_mps_create") : nullptr;
  if (!create) {
    fprintf(stderr, "e3gnn: cannot load %s: %s\n", path.c_str(), dlerror());
    return;
  }
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "e3gnn";
  plugin.info = "SevenNet on the Apple GPU, serial, through embedded Python";
  plugin.author = "SevenNet";
  plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(create);
  plugin.handle = handle;
  reinterpret_cast<lammpsplugin_regfunc>(regfunc)(&plugin, lmp);
}
}
