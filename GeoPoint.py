import math
class GeoPoint():
    def __init__(self, lon, lat, isRadian=False, precision=6):
        if not isRadian:
            lon = lon / 180 * math.pi
            lat = lat / 180 * math.pi
        self.longitude = lon
        self.latitude = lat
        self._precision = precision

    def __repr__(self):
        return f"A geodetic point at longitude {self.lon_deg} and latitude {self.lat_deg}"

    def __setattr__(self, name, value):
        "regulate longitude within [-pi,*pi) and latitude [-pi/2,*pi/2]"
        if name[-3:] == 'deg' or name[-6:] == 'degree':
            value = value / 180 * math.pi
        if name[:3] == 'lon':
            name = 'longitude'
            value = (value+math.pi)%(2*math.pi)-math.pi
        elif name[:3] == 'lat':
            name = 'latitude'
            value = math.acos(math.cos(value+math.pi/2))-math.pi/2
        super().__setattr__(name, value)

    def __getattribute__(self, name) -> float:
        if name[0] == '_' or name[:3] == 'get': return super().__getattribute__(name)
        to_output = 180 / math.pi if (name[-3:] == 'deg' or name[-6:] == 'degree') else 1
        if name[:3] == 'lon': name = 'longitude'
        elif name[:3] == 'lat': name = 'latitude'
        return round(super().__getattribute__(name) * to_output, self._precision)

    def get_lonlat(self, unit: str = 'deg') -> tuple:
        if unit == 'deg' or unit == 'degree':
            return (self.longitude_deg, self.latitude_deg)
        else:
            return (self.longitude, self.latitude)

    def get_latlon(self, unit: str = 'deg') -> tuple:
        if unit == 'deg' or unit == 'degree':
            return (self.latitude_deg, self.longitude_deg)
        else:
            return (self.latitude, self.longitude)

def midpoint(point1, point2):
    # const φ1 = lat1 * Math.PI/180; // φ, λ in radians
    # const φ2 = lat2 * Math.PI/180;
    # const Δφ = (lat2-lat1) * Math.PI/180;
    # const Δλ = (lon2-lon1) * Math.PI/180;
    # const Bx = Math.cos(φ2) * Math.cos(λ2-λ1);
    # const By = Math.cos(φ2) * Math.sin(λ2-λ1);
    # const φ3 = Math.atan2(Math.sin(φ1) + Math.sin(φ2),
    #                     Math.sqrt( (Math.cos(φ1)+Bx)*(Math.cos(φ1)+Bx) + By*By ) );
    # const λ3 = λ1 + Math.atan2(By, Math.cos(φ1) + Bx);
    dLon = point2.lon - point1.lon
    Bx = math.cos(point2.lat) * math.cos(dLon)
    By = math.cos(point2.lat) * math.sin(dLon)
    mlat = math.atan2(math.sin(point1.lat) + math.sin(point2.lat),
                    math.sqrt((math.cos(point1.lat) + Bx) * (math.cos(point1.lat) + Bx) + By * By))
    mlon = point1.lon + math.atan2(By, math.cos(point1.lat) + Bx)
    return GeoPoint(mlon, mlat, isRadian=True)