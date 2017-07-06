%define codename %NAME%
%define _topdir %(echo $PWD)/rpm

Name:           %{codename}
Version:        %VERSION%
Release:        1%{?dist}
Summary:        Monte Carlo eXtreme (MCX)

Group:          Applications/Engineering

License:        GPLv3
URL:            http://mcx.space
Source:         %{name}-%{version}.tar.gz
#Source0:        https://github.com/fangq/mcx/archive/v%{version}.zip
BuildRoot:      %(mktemp -ud %{_tmppath}/%{name}-%{version}-%{release}-XXXXXX)

BuildArch:      x86_64
BuildRequires:  cuda
#Requires:       cuda-cudart

%description
%INFO%

%prep
%setup -q -c %{codename}


%build
cd %{codename}
make bin/%{codename}


%install
rm -rf %{buildroot}

install -m 0755 -d %{buildroot}%{_libdir}
install -m 0755 -d %{buildroot}%{_libdir}/%{codename}
install -m 0755 -p %{codename}/bin/%{codename} %{buildroot}%{_libdir}/%{codename}
install -m 0755 -d %{buildroot}%{_bindir}


%clean
rm -fr %{buildroot}

%files
%{_libdir}/%{codename}/%{codename}
%doc %{codename}/AUTHORS.txt %{codename}/ChangeLog.txt %{codename}/LICENSE.txt %{codename}/README.txt


%changelog
* Tue Jul 04 2017 Qianqian Fang <fangqq at gmail.com>
- Initial RPM release
