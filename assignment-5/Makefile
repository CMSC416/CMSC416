CHARMC      = /software/spack-software/2020.05.14/linux-rhel8-ivybridge/gcc-8.4.0/charmpp-6.10.1-gswoeqljmdxdrnw6u5qspfitq4gr4zyc/mpi-linux-x86_64/bin/charmc
OPTS        = -g -O2

OBJS		= Prefix.o
OBJS_DECL	= prefix.decl.h

all: prefix

prefix: $(OBJS) $(OBJS_DECL)
	$(CHARMC) -o prefix $(OBJS) -language charm++ $(OPTS)

prefix.decl.h: prefix.ci
	$(CHARMC) prefix.ci

Prefix.o: Prefix.C Prefix.h prefix.decl.h
	$(CHARMC) $(OPTS) -o Prefix.o Prefix.C

clean:
	rm -f *.decl.h *.def.h *.o charmrun prefix

